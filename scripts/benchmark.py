#!/usr/bin/env python3
"""
Measure prediction latency and accuracy, and write docs/BENCHMARKS.md.

Everything the README claims about speed comes from this script. Run it after
``cueai-train`` so the accuracy section can read models/metrics.json:

    python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cueai.ml.infer import TrajectoryPredictor  # noqa: E402
from cueai.physics.analytic import predict_endpoint  # noqa: E402
from cueai.physics.constants import ShotParams, TableParams  # noqa: E402
from cueai.physics.simulator import Simulator  # noqa: E402


def time_calls(fn, repeats: int) -> dict[str, float]:
    """Wall-clock statistics for a single call, in milliseconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "n": repeats,
    }


def measure_latency(repeats: int) -> dict[str, dict[str, float]]:
    table = TableParams()
    shot = ShotParams(speed=2.5, angle=0.12, english_x=0.2, english_y=-0.1)
    cue = (0.6, 0.635)
    results: dict[str, dict[str, float]] = {}

    rack_sim = Simulator(table=table, dt=0.001, max_time=15.0)
    results["simulator_full_rack"] = time_calls(
        lambda: rack_sim.simulate_shot(shot, full_rack=True), max(repeats // 200, 3)
    )

    two_ball = Simulator(table=table, dt=0.002, max_time=8.0)
    results["simulator_two_ball"] = time_calls(
        lambda: two_ball.simulate_shot(shot, cue_pos=cue, obj_pos=(1.4, 0.7)),
        max(repeats // 50, 5),
    )

    results["closed_form"] = time_calls(
        lambda: predict_endpoint(shot, cue, table), repeats
    )

    gbm_path = ROOT / "models" / "gbm_baseline.joblib"
    if gbm_path.exists():
        import joblib

        from cueai.ml.features import build_features

        bundle = joblib.load(gbm_path)
        row = build_features(shot, cue, (1.4, 0.7), table)[None, :]
        results["gradient_boosting"] = time_calls(
            lambda: bundle["model"].predict(bundle["scaler"].transform(row)), repeats
        )

    predictor = TrajectoryPredictor(model_dir=ROOT / "models")
    if predictor.ready:
        results["surrogate_" + predictor.backend] = time_calls(
            lambda: predictor.predict_fast(shot, cue, obj_pos=(1.4, 0.7), table=table),
            repeats,
        )
        batch = np.repeat(
            predictor.feature_vector(shot, cue, (1.4, 0.7), table)[None, :], 1024, axis=0
        )
        batched = time_calls(lambda: predictor.residual_batch(batch), max(repeats // 20, 5))
        results["cuenet_batch1024"] = {
            key: value / 1024 if key.endswith("_ms") else value
            for key, value in batched.items()
        }

    reference = results["simulator_full_rack"]["mean_ms"]
    for name, entry in results.items():
        if name != "simulator_full_rack":
            entry["speedup_vs_full_rack"] = reference / entry["mean_ms"]
    return results


def format_duration(milliseconds: float) -> str:
    """Human-readable across the five orders of magnitude this table spans."""
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:,.1f} s"
    if milliseconds >= 1:
        return f"{milliseconds:.1f} ms"
    if milliseconds >= 0.01:
        return f"{milliseconds:.3f} ms"
    return f"{milliseconds * 1000:.1f} us"


def render_markdown(latency: dict, metrics: dict | None) -> str:
    labels = {
        "simulator_full_rack": "Numerical simulator, 16-ball rack",
        "simulator_two_ball": "Numerical simulator, cue + object ball",
        "closed_form": "Closed-form solver (no ML)",
        "gradient_boosting": "Gradient boosting on the same features",
        "surrogate_torch": "Closed form + CueNet residual (PyTorch)",
        "surrogate_onnx": "Closed form + CueNet residual (ONNX Runtime)",
        "cuenet_batch1024": "CueNet forward pass only, batch of 1024 (per shot)",
    }
    lines = [
        "# Benchmarks",
        "",
        "Regenerate with `python scripts/benchmark.py`. Single CPU core, no GPU.",
        "",
        f"- Python {platform.python_version()} on {platform.platform()}",
        f"- Processor: {platform.processor() or 'unknown'}",
        "",
        "## Latency per shot",
        "",
        "| Method | Mean | Median | p95 | Speedup vs full rack |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, entry in latency.items():
        # The batched row times the network alone, so it is not a like-for-like
        # end-to-end prediction and does not get a speedup figure.
        speedup = entry.get("speedup_vs_full_rack") if "batch" not in key else None
        lines.append(
            f"| {labels.get(key, key)} | {format_duration(entry['mean_ms'])} | "
            f"{format_duration(entry['median_ms'])} | {format_duration(entry['p95_ms'])} | "
            f"{f'{speedup:,.0f}x' if speedup else '—'} |"
        )

    if metrics:
        models = metrics["models"]
        lines += [
            "",
            "## Accuracy on held-out shots",
            "",
            f"{metrics['n_test']:,} test shots from a {metrics['n_samples']:,} shot dataset. "
            "Error is the distance between predicted and simulated resting position, "
            "averaged over the cue ball and the object ball.",
            "",
            "| Model | Mean error | p95 | Cue ball | Object ball | R² |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        names = {
            "analytic": "Closed form (no fitting)",
            "gbm": "Gradient boosting on raw features",
            "cuenet": "Closed form + CueNet residual",
        }
        for key, label in names.items():
            m = models[key]
            lines.append(
                f"| {label} | {m['euclidean_mm']:.0f} mm | {m['p95_mm']:.0f} mm | "
                f"{m['cue_mm']:.0f} mm | {m['obj_mm']:.0f} mm | {m['r2']:.3f} |"
            )
        lines += [
            "",
            "## Where prediction stops working",
            "",
            "Resting position is a smooth function of the shot until the ball starts "
            "ricocheting between cushions. Mean error in mm by cushion contacts:",
            "",
            "| Cushion contacts | Shots | Closed form | Gradient boosting | CueNet residual |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in metrics.get("by_cushion_contacts", []):
            lines.append(
                f"| {row['cushion_contacts']} | {row['n']} | {row['analytic']:.0f} | "
                f"{row['gbm']:.0f} | {row['cuenet']:.0f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--out", type=str, default="docs/BENCHMARKS.md")
    args = parser.parse_args()

    latency = measure_latency(args.repeats)
    metrics_path = ROOT / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    (ROOT / "models" / "latency.json").write_text(json.dumps(latency, indent=2) + "\n")
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(latency, metrics))
    print(json.dumps(latency, indent=2))
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
