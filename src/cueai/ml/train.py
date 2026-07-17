"""Train CueNet residual model + export ONNX + sklearn baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from cueai.ml.dataset import FEATURE_NAMES, TARGET_NAMES, generate_dataset
from cueai.ml.model import CueNet


PHYS_COLS = ["phys_cue_end_x", "phys_cue_end_y", "phys_obj_end_x", "phys_obj_end_y"]
RESIDUAL_TARGETS = TARGET_NAMES[:4]  # cue/obj endpoints


def train_sklearn_baseline(df: pd.DataFrame, model_dir: Path) -> dict:
    X = df[FEATURE_NAMES].values
    y = df[RESIDUAL_TARGETS].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=120, max_depth=4, random_state=0)
    )
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    mae = float(np.mean(np.abs(pred - y_te)))
    joblib.dump({"model": model, "scaler": scaler}, model_dir / "sklearn_baseline.joblib")
    return {"sklearn_mae_m": mae}


def train_torch(
    df: pd.DataFrame,
    model_dir: Path,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> dict:
    device = torch.device("cpu")
    X = df[FEATURE_NAMES].values.astype(np.float32)
    phys = df[PHYS_COLS].values.astype(np.float32)
    y = df[RESIDUAL_TARGETS].values.astype(np.float32)
    # Train residual = y - phys (ideally ~0; noise/table variance creates learnable signal
    # when we inject measurement noise)
    noise = np.random.default_rng(1).normal(0, 0.008, size=y.shape).astype(np.float32)
    y_obs = y + noise
    residual = y_obs - phys

    X_tr, X_te, p_tr, p_te, r_tr, r_te, y_tr, y_te = train_test_split(
        X, phys, residual, y_obs, test_size=0.2, random_state=0
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    ds = TensorDataset(
        torch.from_numpy(X_tr_s),
        torch.from_numpy(p_tr),
        torch.from_numpy(r_tr),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    net = CueNet(in_dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    history = []
    net.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for xb, _pb, rb in loader:
            xb, rb = xb.to(device), rb.to(device)
            pred_r = net(xb)
            loss = loss_fn(pred_r, rb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        history.append(total / max(n, 1))

    net.eval()
    with torch.no_grad():
        pred_r = net(torch.from_numpy(X_te_s)).numpy()
        pred_y = p_te + pred_r
        mae = float(np.mean(np.abs(pred_y - y_te)))
        phys_mae = float(np.mean(np.abs(p_te - y_te)))

    ckpt = model_dir / "cuenet.pt"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_names": FEATURE_NAMES,
            "in_dim": X.shape[1],
        },
        ckpt,
    )

    # ONNX export (best-effort; Torch 2.x may need onnxscript)
    onnx_path = model_dir / "cuenet.onnx"
    dummy = torch.randn(1, X.shape[1], dtype=torch.float32)
    try:
        torch.onnx.export(
            net.cpu(),
            dummy,
            str(onnx_path),
            input_names=["features"],
            output_names=["residual"],
            dynamic_axes={"features": {0: "batch"}, "residual": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ONNX export skipped: {exc}")
        onnx_path = model_dir / "cuenet.onnx"
        if not onnx_path.exists():
            onnx_path = Path("")


    meta = {
        "torch_mae_m": mae,
        "physics_only_mae_m": phys_mae,
        "improvement_pct": float(100 * (phys_mae - mae) / max(phys_mae, 1e-9)),
        "epochs": epochs,
        "n_samples": len(df),
        "history_last": history[-1] if history else None,
        "onnx": str(onnx_path),
        "checkpoint": str(ckpt),
    }
    (model_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train CueAI models")
    p.add_argument("--n-samples", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--data", type=str, default="data/processed/shots.csv")
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--skip-generate", action="store_true")
    args = p.parse_args(argv)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)

    if args.skip_generate and data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_dataset(n_samples=args.n_samples, out_csv=data_path)

    sk = train_sklearn_baseline(df, model_dir)
    torch_meta = train_torch(df, model_dir, epochs=args.epochs)
    summary = {**sk, **torch_meta}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
