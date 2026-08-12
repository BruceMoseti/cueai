"""
Synthetic dataset generation from the physics simulator.

Each row pairs a shot with two things: the high-fidelity simulator outcome
(the label) and the closed-form analytic prediction for the same shot (the
baseline). The learned model predicts the gap between them, so its value is
measured as error reduction over a real physical baseline rather than over
nothing.

Two properties are worth noting:

* Table friction and restitution are resampled per shot (domain randomisation),
  which is what makes the residual a function of the inputs rather than noise.
* Each sample draws from its own seeded generator, so the dataset is identical
  whether it is produced on 1 core or 32.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pocket.physics.analytic import predict_endpoint
from pocket.physics.constants import ShotParams, TableParams
from pocket.physics.simulator import FEATURE_NAMES, Simulator, shot_feature_vector

TARGET_NAMES = [
    "cue_end_x",
    "cue_end_y",
    "obj_end_x",
    "obj_end_y",
    "path_length_cue",
    "max_speed_cue",
]

# Closed-form prediction for the same four endpoints, used as the residual base.
BASELINE_NAMES = [
    "baseline_cue_end_x",
    "baseline_cue_end_y",
    "baseline_obj_end_x",
    "baseline_obj_end_y",
]

# Outcome complexity, recorded for analysis. These are results, not inputs, so
# they must never be used as model features.
DIAGNOSTIC_NAMES = ["n_cushion", "n_collision", "potted"]

# Practical cue tip offset limit; beyond ~0.5R a real stroke miscues.
MAX_TIP_OFFSET = 0.5


def _path_length(traj: np.ndarray) -> float:
    if len(traj) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def simulate_sample(index: int, seed: int = 42) -> dict[str, float]:
    """
    Simulate one random shot.

    The generator is keyed on (seed, index) so the sample is reproducible in
    isolation, independent of how the work is distributed across processes.
    """
    rng = np.random.default_rng([seed, index])

    table = TableParams(
        mu_slide=float(rng.uniform(0.15, 0.28)),
        mu_roll=float(rng.uniform(0.006, 0.018)),
        mu_spin=float(rng.uniform(0.03, 0.06)),
        e_cushion=float(rng.uniform(0.75, 0.92)),
        friction_noise_amp=float(rng.uniform(0.0, 0.04)),
    )
    sim = Simulator(table=table, dt=0.002, max_time=8.0)
    length, width = table.length, table.width
    radius = sim.ball_params.radius

    cue_pos = np.array(
        [rng.uniform(radius * 3, length * 0.4), rng.uniform(radius * 3, width - radius * 3)]
    )
    obj_pos = np.array(
        [
            rng.uniform(length * 0.45, length - radius * 3),
            rng.uniform(radius * 3, width - radius * 3),
        ]
    )
    if np.linalg.norm(cue_pos - obj_pos) < 4 * radius:
        obj_pos[0] = min(length - 3 * radius, cue_pos[0] + 0.35)

    shot = ShotParams(
        speed=float(rng.uniform(0.5, 6.0)),
        angle=float(rng.uniform(-np.pi, np.pi)),
        english_x=float(rng.uniform(-MAX_TIP_OFFSET, MAX_TIP_OFFSET)),
        english_y=float(rng.uniform(-MAX_TIP_OFFSET, MAX_TIP_OFFSET)),
        cue_elevation=float(rng.uniform(0.0, 0.15)),
    )
    # Slightly over half the shots are aimed at the object ball, so the dataset
    # is not dominated by shots that never make contact.
    if rng.random() < 0.55:
        delta = obj_pos - cue_pos
        shot.angle = float(np.arctan2(delta[1], delta[0]) + rng.normal(0, 0.08))

    result = sim.simulate_shot(
        shot,
        cue_pos=(float(cue_pos[0]), float(cue_pos[1])),
        obj_pos=(float(obj_pos[0]), float(obj_pos[1])),
    )

    features = shot_feature_vector(shot, cue_pos, obj_pos, table)
    cue_vel = result.velocities[0]
    obj_end = result.endpoints.get(1, np.zeros(2))
    targets = [
        float(result.endpoints[0][0]),
        float(result.endpoints[0][1]),
        float(obj_end[0]),
        float(obj_end[1]),
        _path_length(result.trajectories[0]),
        float(np.max(np.linalg.norm(cue_vel, axis=1))) if len(cue_vel) else 0.0,
    ]

    row = {name: float(value) for name, value in zip(FEATURE_NAMES, features)}
    row.update(dict(zip(TARGET_NAMES, targets)))

    # Closed-form baseline: analytic stopping point for the cue ball, and the
    # object ball left where it started, since the analytic model has no notion
    # of ball-ball contact.
    baseline_cue = predict_endpoint(shot, cue_pos, table, radius=radius)
    row["baseline_cue_end_x"] = float(baseline_cue[0])
    row["baseline_cue_end_y"] = float(baseline_cue[1])
    row["baseline_obj_end_x"] = float(obj_pos[0])
    row["baseline_obj_end_y"] = float(obj_pos[1])

    row["n_cushion"] = float(result.cushion_events)
    row["n_collision"] = float(result.collision_events)
    row["potted"] = float(sum(1 for was_potted in result.pocketed.values() if was_potted))
    return row


def generate_dataset(
    n_samples: int = 4000,
    seed: int = 42,
    out_csv: str | Path | None = "data/processed/shots.csv",
    jobs: int | None = None,
) -> pd.DataFrame:
    """Simulate ``n_samples`` random shots, in parallel by default."""
    jobs = jobs or min(os.cpu_count() or 1, 16)
    indices = range(n_samples)
    progress = {"total": n_samples, "desc": "Simulating shots", "unit": "shot"}

    if jobs <= 1:
        rows = [simulate_sample(i, seed) for i in tqdm(indices, **progress)]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = pool.map(simulate_sample, indices, [seed] * n_samples, chunksize=8)
            rows = list(tqdm(futures, **progress))

    df = pd.DataFrame(rows)
    if out_csv is not None:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    generate_dataset(500)
