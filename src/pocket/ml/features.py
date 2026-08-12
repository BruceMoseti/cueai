"""
Feature construction, shared by training and serving.

The raw shot parameters alone are a poor input for a residual model: to know
where the closed-form prediction went wrong, the network would first have to
rediscover cushion reflection geometry from a speed and an angle. So the
features also carry what the closed-form solver already worked out — where it
thinks the ball stops, how many cushions it expects, whether it expects a pot —
plus the ghost-ball geometry that decides whether the object ball is touched at
all.

Two consequences worth stating:

* The predicted cushion count tells the model when the outcome is chaotic, so it
  can fall back to "trust the physics" instead of guessing.
* Training and serving must build features identically. Both paths go through
  :func:`build_features`, and ``tests/test_features.py`` pins them together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pocket.physics.analytic import solve_free_ball
from pocket.physics.constants import BallParams, ShotParams, TableParams
from pocket.physics.simulator import FEATURE_NAMES as SHOT_FEATURE_NAMES

BASELINE_FEATURE_NAMES = [
    "base_cue_x",        # closed-form resting position of the cue ball
    "base_cue_y",
    "base_cushions",     # cushion contacts the closed-form solver expects
    "base_potted",       # whether it expects the cue ball to drop
    "base_travel",       # straight-line distance from cue start to that endpoint
    "contact_along",     # object ball distance projected onto the aim line
    "contact_perp",      # perpendicular miss distance of the aim line, signed
    "will_contact",      # aim line passes within one ball diameter, ahead of the cue
]

FEATURE_NAMES = SHOT_FEATURE_NAMES + BASELINE_FEATURE_NAMES


def _spin_contact(shot: ShotParams, radius: float) -> np.ndarray:
    """ω × (-Rẑ), the spin contribution to contact-point velocity."""
    omega = shot.initial_omega(BallParams(radius=radius))
    return np.array([-radius * omega[1], radius * omega[0]])


def build_features(
    shot: ShotParams,
    cue_pos: np.ndarray | tuple[float, float],
    obj_pos: np.ndarray | tuple[float, float] | None,
    table: TableParams,
    radius: float = 0.028575,
) -> np.ndarray:
    """Model input for a single shot. Order matches :data:`FEATURE_NAMES`."""
    from pocket.physics.simulator import shot_feature_vector

    cue = np.asarray(cue_pos, dtype=np.float64)
    obj = np.asarray(obj_pos, dtype=np.float64) if obj_pos is not None else None

    outcome = solve_free_ball(
        cue,
        shot.speed * np.array([np.cos(shot.angle), np.sin(shot.angle)]),
        _spin_contact(shot, radius),
        table,
        radius,
    )

    aim = np.array([np.cos(shot.angle), np.sin(shot.angle)])
    normal = np.array([-aim[1], aim[0]])
    if obj is None:
        along, perp = 0.0, 0.0
    else:
        offset = obj - cue
        along, perp = float(offset @ aim), float(offset @ normal)
    will_contact = float(along > 0 and abs(perp) < 2 * radius)

    return np.concatenate(
        [
            shot_feature_vector(shot, cue, obj, table),
            [
                outcome.position[0],
                outcome.position[1],
                float(outcome.cushions),
                float(outcome.potted),
                float(np.linalg.norm(outcome.position - cue)),
                along,
                perp,
                will_contact,
            ],
        ]
    )


def build_feature_frame(df: pd.DataFrame, radius: float = 0.028575) -> np.ndarray:
    """
    Rebuild the feature matrix from a generated dataset.

    The stored CSV keeps raw shot and table parameters rather than derived
    features, so feature engineering can change without re-running 20,000
    simulations.
    """
    rows = []
    for record in df.to_dict("records"):
        shot = ShotParams(
            speed=float(record["speed"]),
            angle=float(record["angle"]),
            english_x=float(record["english_x"]),
            english_y=float(record["english_y"]),
            cue_elevation=float(record["cue_elevation"]),
        )
        table = TableParams(
            mu_slide=float(record["mu_slide"]),
            mu_roll=float(record["mu_roll"]),
            mu_spin=float(record["mu_spin"]),
            e_cushion=float(record["e_cushion"]),
            friction_noise_amp=float(record["friction_noise_amp"]),
        )
        rows.append(
            build_features(
                shot,
                (float(record["cue_x"]), float(record["cue_y"])),
                (float(record["obj_x"]), float(record["obj_y"])),
                table,
                radius,
            )
        )
    return np.asarray(rows, dtype=np.float64)
