#!/usr/bin/env python3
"""
Render the figures used in the README, headlessly.

    python scripts/make_figures.py

Writes docs/assets/*.png. The validation and break figures need nothing but the
physics; the accuracy and latency figures read models/metrics.json and
models/latency.json, so run training and scripts/benchmark.py first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocket.physics import analytic  # noqa: E402
from pocket.physics.ball import Ball, MotionState, integrate_ball  # noqa: E402
from pocket.physics.constants import BallParams, ShotParams, TableParams  # noqa: E402
from pocket.physics.simulator import Simulator  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
CLOTH = "#12764a"
INK = "#1b1f23"
ACCENT = "#e06c1f"
BLUE = "#2f6fb0"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    }
)


def _roll_to_transition(v0: float, table: TableParams) -> tuple[float, float]:
    """Integrate a centre-ball hit up to the sliding/rolling transition."""
    params = BallParams()
    ball = Ball(
        id=0,
        pos=np.array([0.0, 0.635]),
        vel=np.array([v0, 0.0]),
        omega=np.zeros(3),
        params=params,
    )
    for _ in range(400_000):
        ball = integrate_ball(ball, table, 1e-4)
        if ball.motion_state(table) is MotionState.ROLLING:
            return ball.speed(), float(ball.pos[0])
    raise RuntimeError("ball never reached the rolling phase")


def figure_validation() -> Path:
    """Simulator versus closed form for the sliding phase."""
    table = TableParams(friction_noise_amp=0.0)
    speeds = np.linspace(0.6, 4.0, 12)
    measured = np.array([_roll_to_transition(float(v), table) for v in speeds])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))

    axes[0].plot(speeds, analytic.ROLLING_SPEED_RATIO * speeds, "-", color=INK, lw=1.4,
                 label=r"closed form  $\frac{5}{7}v_0$")
    axes[0].plot(speeds, measured[:, 0], "o", color=ACCENT, ms=4.5, label="simulator")
    axes[0].set_xlabel("launch speed $v_0$  (m/s)")
    axes[0].set_ylabel("speed at rolling  (m/s)")
    axes[0].set_title("Sliding ends at 5/7 of launch speed")
    axes[0].legend(frameon=False)

    reference = [analytic.slide_distance(float(v), table.mu_slide) for v in speeds]
    axes[1].plot(speeds, reference, "-", color=INK, lw=1.4,
                 label=r"closed form  $\frac{12v_0^2}{49\mu_s g}$")
    axes[1].plot(speeds, measured[:, 1], "o", color=BLUE, ms=4.5, label="simulator")
    axes[1].set_xlabel("launch speed $v_0$  (m/s)")
    axes[1].set_ylabel("sliding distance  (m)")
    axes[1].set_title("Sliding distance matches theory")
    axes[1].legend(frameon=False)

    worst = float(
        np.max(np.abs(measured[:, 1] - reference) / np.maximum(reference, 1e-9)) * 100
    )
    fig.suptitle(
        f"Physics validation: worst-case deviation from closed form {worst:.1f}%",
        fontsize=10,
    )
    return _save(fig, "validation.png")


def _draw_table(ax, table: TableParams) -> None:
    ax.add_patch(
        plt.Rectangle((0, 0), table.length, table.width, facecolor=CLOTH, edgecolor="#5c3a1e", lw=6)
    )
    for x, y in table.pockets:
        ax.add_patch(plt.Circle((x, y), table.pocket_radius, color="#0b0b0b", zorder=3))
    ax.set_xlim(-0.08, table.length + 0.08)
    ax.set_ylim(-0.08, table.width + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")


def figure_break() -> Path:
    """A full 16-ball break, which is what the simulator is actually solving."""
    table = TableParams(friction_noise_amp=0.02)
    sim = Simulator(table=table, dt=0.001, max_time=15.0, collision_passes=20)
    shot = ShotParams(speed=6.5, angle=0.008, english_y=0.25)
    result = sim.simulate_shot(shot, full_rack=True, seed=7)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    _draw_table(ax, table)
    for ball_id, path in result.trajectories.items():
        visible = path[np.all(path >= 0, axis=1)]
        if len(visible) < 2:
            continue
        meta = result.ball_meta.get(ball_id, {})
        colour = np.array(meta.get("color", [220, 220, 220])) / 255
        ax.plot(visible[:, 0], visible[:, 1], "-", color=colour, lw=1.6, alpha=0.9, zorder=4)
        ax.plot(visible[-1, 0], visible[-1, 1], "o", color=colour, ms=6,
                markeredgecolor="#111", markeredgewidth=0.6, zorder=5)

    potted = sum(1 for was_potted in result.pocketed.values() if was_potted)
    ax.set_title(
        f"16-ball break: {result.collision_events} ball-ball contacts, "
        f"{result.cushion_events} cushion contacts, {potted} potted",
        fontsize=10,
    )
    return _save(fig, "break_shot.png")


def figure_spin() -> Path:
    """Draw, stun and follow from the same stroke speed."""
    table = TableParams(friction_noise_amp=0.0)
    sim = Simulator(table=table, dt=5e-4, max_time=12.0, collision_passes=10)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    _draw_table(ax, table)

    # One lane per stroke so the three cue-ball paths do not overlap.
    # The gap is kept short so the centre-ball shot is still sliding at contact,
    # which is what makes the three strokes visibly different.
    start_x = 0.95
    lanes = [
        (0.95, -0.45, "backspin  (tip 0.45R below centre)  ->  draw", ACCENT),
        (0.635, 0.0, "centre ball  ->  stun", "#f4f4f4"),
        (0.32, 0.45, "topspin  (tip 0.45R above centre)  ->  follow", BLUE),
    ]
    for lane_y, english_y, label, colour in lanes:
        result = sim.simulate_shot(
            ShotParams(speed=2.5, angle=0.0, english_y=english_y),
            cue_pos=(start_x, lane_y),
            obj_pos=(1.2, lane_y),
            full_rack=False,
        )
        cue = result.trajectories[0]
        ax.plot(cue[:, 0], cue[:, 1], "-", color=colour, lw=2.2, zorder=4)
        ax.plot(start_x, lane_y, "o", color=colour, ms=9, markeredgecolor="#111", zorder=5)
        ax.plot(cue[-1, 0], cue[-1, 1], "*", color=colour, ms=15, markeredgecolor="#111",
                markeredgewidth=0.5, zorder=6)
        ax.plot(1.2, lane_y, "o", color="#e8e2b0", ms=9, markeredgecolor="#111", zorder=5)
        ax.text(0.10, lane_y + 0.09, label, color=colour, fontsize=8.5, zorder=7)
        ax.text(float(cue[-1, 0]), lane_y - 0.13, f"{float(cue[-1, 0]):.2f} m",
                color=colour, fontsize=8, ha="center", zorder=7)

    ax.annotate(
        "contact",
        xy=(1.2, 0.05),
        xytext=(1.2, -0.02),
        color="#f4f4f4",
        fontsize=8,
        ha="center",
    )
    ax.axvline(1.2, color="#f4f4f4", alpha=0.25, lw=0.8, ls=":", zorder=2)
    ax.set_title(
        "Same 2.5 m/s stroke, three tip heights: the cue ball draws back, stops short, "
        "or follows through\n(circles = start, stars = resting position)",
        fontsize=9.5,
    )
    return _save(fig, "spin_control.png")


def figure_accuracy(metrics: dict) -> Path:
    """Error by cushion contacts, which is where the approach runs out."""
    rows = metrics["by_cushion_contacts"]
    labels = [row["cushion_contacts"] for row in rows]
    series = [
        ("closed form", "analytic", "#9aa4ad"),
        ("gradient boosting", "gbm", BLUE),
        ("closed form + CueNet", "cuenet", ACCENT),
    ]
    positions = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for offset, (label, key, colour) in enumerate(series):
        values = [row[key] for row in rows]
        ax.bar(positions + (offset - 1) * width, values, width, label=label, color=colour)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{label}\n(n={row['n']})" for label, row in zip(labels, rows)])
    ax.set_xlabel("cushion contacts before coming to rest")
    ax.set_ylabel("mean endpoint error  (mm)")
    ax.set_title(
        "Prediction error grows with every cushion contact, until the outcome is chaotic",
        fontsize=10,
    )
    ax.legend(frameon=False)
    return _save(fig, "accuracy.png")


def figure_reliability(metrics: dict) -> Path:
    """Error against coverage, when the closed-form cushion count is used as a gate."""
    rows = metrics["risk_coverage"]
    coverage = [row["coverage_pct"] for row in rows]
    series = [
        ("closed form", "analytic", "#9aa4ad", "o"),
        ("gradient boosting", "gbm", BLUE, "s"),
        ("closed form + CueNet", "cuenet", ACCENT, "D"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for label, key, colour, marker in series:
        ax.plot(coverage, [row[key] for row in rows], marker + "-", color=colour,
                lw=1.6, ms=5, label=label)
    for row in rows:
        gate = row["expected_cushions_at_most"]
        ax.annotate(
            "no gate" if gate == "all" else f"≤{gate}",
            xy=(row["coverage_pct"], row["cuenet"]),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=ACCENT,
        )
    ax.set_xlabel("shots answered by the fast path  (%)")
    ax.set_ylabel("mean endpoint error  (mm)")
    ax.set_title(
        "Choosing how much to answer: the closed-form cushion count is a free gate",
        fontsize=10,
    )
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, "reliability.png")


def figure_latency(latency: dict) -> Path:
    """Cost per shot, log scale, because the range spans five orders of magnitude."""
    labels = {
        "simulator_full_rack": "simulator, 16 balls",
        "simulator_two_ball": "simulator, 2 balls",
        "closed_form": "closed form",
        "surrogate_onnx": "closed form + CueNet (ONNX)",
        "surrogate_torch": "closed form + CueNet (PyTorch)",
        "cuenet_batch1024": "CueNet forward pass, batched",
    }
    entries = [
        (labels[key], value["mean_ms"]) for key, value in latency.items() if key in labels
    ]
    entries.sort(key=lambda item: item[1])

    def label_time(milliseconds: float) -> str:
        if milliseconds >= 1000:
            return f"{milliseconds / 1000:,.1f} s"
        if milliseconds >= 1:
            return f"{milliseconds:.0f} ms"
        if milliseconds >= 0.01:
            return f"{milliseconds:.2f} ms"
        return f"{milliseconds * 1000:.1f} us"

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    names = [name for name, _ in entries]
    values = [value for _, value in entries]
    colours = [ACCENT if "CueNet" in name or "closed" in name else "#9aa4ad" for name in names]
    ax.barh(names, values, color=colours)
    ax.set_xscale("log")
    ax.set_xlabel("time per shot, log scale")
    for index, value in enumerate(values):
        ax.text(value * 1.15, index, label_time(value), va="center", fontsize=8)
    ax.set_xlim(min(values) * 0.5, max(values) * 6)
    ax.set_title("Cost of one shot prediction", fontsize=10)
    ax.grid(axis="y", visible=False)
    return _save(fig, "latency.png")


def _save(fig, name: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def main() -> None:
    figure_validation()
    figure_spin()
    figure_break()

    metrics_path = ROOT / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if metrics.get("by_cushion_contacts"):
        figure_accuracy(metrics)
    else:
        print("skipping accuracy figure: run training to produce models/metrics.json")

    if metrics.get("risk_coverage"):
        figure_reliability(metrics)
    else:
        print("skipping reliability figure: run training to produce models/metrics.json")

    latency_path = ROOT / "models" / "latency.json"
    if latency_path.exists():
        figure_latency(json.loads(latency_path.read_text()))
    else:
        print("skipping latency figure: models/latency.json missing")


if __name__ == "__main__":
    main()
