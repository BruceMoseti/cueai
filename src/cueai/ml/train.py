"""
Train the CueNet residual model, export ONNX, and score it against baselines.

The learned quantity is the gap between a closed-form prediction and the
high-fidelity simulator, so every reported number is an error reduction over a
physical baseline. Three models are compared on the same held-out split:

    analytic   closed-form, no fitting            (the baseline to beat)
    gbm        gradient boosting on raw features  (does ML help at all?)
    cuenet     MLP predicting the analytic residual

Results land in ``models/metrics.json``, which is what the README quotes.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from cueai.ml.dataset import BASELINE_NAMES, FEATURE_NAMES, TARGET_NAMES, generate_dataset
from cueai.ml.model import CueNet

# Endpoint targets, in metres: (cue_x, cue_y, obj_x, obj_y)
ENDPOINT_TARGETS = TARGET_NAMES[:4]
TEST_SIZE = 0.2
SPLIT_SEED = 0


def endpoint_errors(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """
    Error summary in millimetres.

    ``euclidean`` is the mean distance between predicted and true resting
    positions, averaged over the cue and object ball, which is the number a
    player or a downstream planner actually cares about.
    """
    per_axis = np.abs(pred - truth)
    cue_dist = np.linalg.norm(pred[:, :2] - truth[:, :2], axis=1)
    obj_dist = np.linalg.norm(pred[:, 2:] - truth[:, 2:], axis=1)
    both = np.concatenate([cue_dist, obj_dist])
    ss_res = float(np.sum((pred - truth) ** 2))
    ss_tot = float(np.sum((truth - truth.mean(axis=0)) ** 2))
    return {
        "mae_mm": float(per_axis.mean() * 1000),
        "euclidean_mm": float(both.mean() * 1000),
        "p95_mm": float(np.percentile(both, 95) * 1000),
        "cue_mm": float(cue_dist.mean() * 1000),
        "obj_mm": float(obj_dist.mean() * 1000),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def split_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """One split shared by every model, so the comparison is apples to apples."""
    return train_test_split(
        np.arange(n_rows), test_size=TEST_SIZE, random_state=SPLIT_SEED
    )


def stratify_by_contacts(
    predictions: dict[str, np.ndarray], truth: np.ndarray, n_cushion: np.ndarray
) -> list[dict]:
    """
    Break accuracy down by how many cushions the cue ball touched.

    Resting position is a smooth function of the shot until the ball starts
    ricocheting; past a couple of cushion contacts the outcome is chaotic and no
    model should be expected to predict it. Reporting the breakdown states the
    limit of the approach instead of hiding it in an average.
    """
    buckets = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3+", 3, 10_000)]
    rows = []
    for label, low, high in buckets:
        mask = (n_cushion >= low) & (n_cushion <= high)
        if not mask.any():
            continue
        row: dict = {"cushion_contacts": label, "n": int(mask.sum())}
        for name, pred in predictions.items():
            row[name] = endpoint_errors(pred[mask], truth[mask])["euclidean_mm"]
        rows.append(row)
    return rows


def train_gbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_dir: Path,
) -> tuple[np.ndarray, dict[str, float]]:
    scaler = StandardScaler()
    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0)
    )
    model.fit(scaler.fit_transform(x_train), y_train)
    pred = model.predict(scaler.transform(x_test))
    joblib.dump({"model": model, "scaler": scaler}, model_dir / "gbm_baseline.joblib")
    return pred, endpoint_errors(pred, y_test)


def train_cuenet(
    x_train: np.ndarray,
    base_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    base_test: np.ndarray,
    y_test: np.ndarray,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float], list[float]]:
    torch.manual_seed(seed)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_test_s = scaler.transform(x_test).astype(np.float32)

    # The network only has to explain what the closed-form model misses:
    # cushion energy loss, ball-ball contact, throw, and pocketing.
    residual_train = y_train - base_train

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_s), torch.from_numpy(residual_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    net = CueNet(in_dim=x_train.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    history: list[float] = []
    for _ in range(epochs):
        net.train()
        running, count = 0.0, 0
        for features, residual in loader:
            loss = loss_fn(net(features), residual)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(features)
            count += len(features)
        sched.step()
        history.append(running / max(count, 1))

    net.eval()
    with torch.no_grad():
        pred = base_test + net(torch.from_numpy(x_test_s)).numpy()

    torch.save(
        {
            "state_dict": net.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_names": FEATURE_NAMES,
            "in_dim": x_train.shape[1],
        },
        model_dir / "cuenet.pt",
    )
    _export_onnx(net, x_train.shape[1], model_dir / "cuenet.onnx")
    return pred, endpoint_errors(pred, y_test), history


def _export_onnx(net: CueNet, in_dim: int, path: Path) -> bool:
    """Export for runtime-agnostic serving. Non-fatal: the .pt checkpoint stands alone."""
    try:
        torch.onnx.export(
            net.cpu(),
            (torch.randn(1, in_dim, dtype=torch.float32),),
            str(path),
            input_names=["features"],
            output_names=["residual"],
            dynamic_axes={"features": {0: "batch"}, "residual": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        return True
    except Exception as exc:
        print(f"ONNX export skipped: {exc}")
        return False


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train CueAI models")
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=str, default="data/processed/shots.csv")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)

    if args.skip_generate and data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_dataset(n_samples=args.n_samples, out_csv=data_path)

    features = df[FEATURE_NAMES].to_numpy(np.float32)
    baselines = df[BASELINE_NAMES].to_numpy(np.float32)
    targets = df[ENDPOINT_TARGETS].to_numpy(np.float32)
    train_idx, test_idx = split_indices(len(df))
    x_train, x_test = features[train_idx], features[test_idx]
    base_train, base_test = baselines[train_idx], baselines[test_idx]
    y_train, y_test = targets[train_idx], targets[test_idx]

    analytic = endpoint_errors(base_test, y_test)
    gbm_pred, gbm = train_gbm(x_train, y_train, x_test, y_test, model_dir)
    cuenet_pred, cuenet, history = train_cuenet(
        x_train,
        base_train,
        y_train,
        x_test,
        base_test,
        y_test,
        model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    predictions = {"analytic": base_test, "gbm": gbm_pred, "cuenet": cuenet_pred}
    metrics = {
        "n_samples": len(df),
        "n_test": len(y_test),
        "epochs": args.epochs,
        "final_train_loss": history[-1] if history else None,
        "models": {"analytic": analytic, "gbm": gbm, "cuenet": cuenet},
        "by_cushion_contacts": stratify_by_contacts(
            predictions, y_test, df["n_cushion"].to_numpy()[test_idx]
        ),
        "error_reduction_vs_analytic_pct": round(
            100 * (1 - cuenet["euclidean_mm"] / analytic["euclidean_mm"]), 1
        ),
        "error_reduction_vs_gbm_pct": round(
            100 * (1 - cuenet["euclidean_mm"] / gbm["euclidean_mm"]), 1
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
