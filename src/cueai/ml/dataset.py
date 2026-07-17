"""Synthetic dataset generation from the physics simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cueai.physics.constants import ShotParams, TableParams
from cueai.physics.simulator import FEATURE_NAMES, Simulator, shot_feature_vector


TARGET_NAMES = [
    "cue_end_x",
    "cue_end_y",
    "obj_end_x",
    "obj_end_y",
    "path_length_cue",
    "max_speed_cue",
]


def _path_length(traj: np.ndarray) -> float:
    if len(traj) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def generate_dataset(
    n_samples: int = 4000,
    seed: int = 42,
    out_csv: str | Path | None = "data/processed/shots.csv",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for _ in tqdm(range(n_samples), desc="Generating shots"):
        table = TableParams(
            mu_slide=float(rng.uniform(0.15, 0.28)),
            mu_roll=float(rng.uniform(0.006, 0.018)),
            mu_spin=float(rng.uniform(0.03, 0.06)),
            e_cushion=float(rng.uniform(0.75, 0.92)),
            friction_noise_amp=float(rng.uniform(0.0, 0.04)),
        )
        sim = Simulator(table=table, dt=0.002, max_time=8.0)
        L, W = table.length, table.width
        R = sim.ball_params.radius
        cue_pos = np.array(
            [rng.uniform(R * 3, L * 0.4), rng.uniform(R * 3, W - R * 3)],
            dtype=np.float64,
        )
        obj_pos = np.array(
            [rng.uniform(L * 0.45, L - R * 3), rng.uniform(R * 3, W - R * 3)],
            dtype=np.float64,
        )
        # Avoid overlapping start
        if np.linalg.norm(cue_pos - obj_pos) < 4 * R:
            obj_pos[0] = min(L - 3 * R, cue_pos[0] + 0.35)

        shot = ShotParams(
            speed=float(rng.uniform(0.5, 6.0)),
            angle=float(rng.uniform(-np.pi, np.pi)),
            english_x=float(rng.uniform(-0.6, 0.6)),
            english_y=float(rng.uniform(-0.6, 0.6)),
            cue_elevation=float(rng.uniform(0.0, 0.15)),
        )

        # Aim somewhat toward object ball sometimes
        if rng.random() < 0.55:
            delta = obj_pos - cue_pos
            shot.angle = float(np.arctan2(delta[1], delta[0]) + rng.normal(0, 0.08))

        result = sim.simulate_shot(
            shot,
            cue_pos=(float(cue_pos[0]), float(cue_pos[1])),
            obj_pos=(float(obj_pos[0]), float(obj_pos[1])),
        )
        x = shot_feature_vector(shot, cue_pos, obj_pos, table)
        cue_traj = result.trajectories[0]
        obj_traj = result.trajectories.get(1, np.zeros((1, 2)))
        cue_vel = result.velocities[0]
        y = np.array(
            [
                result.endpoints[0][0],
                result.endpoints[0][1],
                result.endpoints.get(1, np.zeros(2))[0],
                result.endpoints.get(1, np.zeros(2))[1],
                _path_length(cue_traj),
                float(np.max(np.linalg.norm(cue_vel, axis=1))) if len(cue_vel) else 0.0,
            ],
            dtype=np.float64,
        )
        # Physics baseline = endpoints without ML (same for residual training target)
        row = {n: float(v) for n, v in zip(FEATURE_NAMES, x)}
        for n, v in zip(TARGET_NAMES, y):
            row[n] = float(v)
        # Physics-only endpoint prediction used as baseline inside residual model
        row["phys_cue_end_x"] = float(result.endpoints[0][0])
        row["phys_cue_end_y"] = float(result.endpoints[0][1])
        row["phys_obj_end_x"] = float(result.endpoints.get(1, np.zeros(2))[0])
        row["phys_obj_end_y"] = float(result.endpoints.get(1, np.zeros(2))[1])
        rows.append(row)

    df = pd.DataFrame(rows)
    if out_csv is not None:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    generate_dataset(500)
