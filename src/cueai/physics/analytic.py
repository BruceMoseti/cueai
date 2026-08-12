"""
Closed-form solutions for the straight-shot limit of the cloth model.

These are the textbook results the numerical integrator in :mod:`cueai.physics.ball`
must reproduce, so they serve two purposes:

1. Ground truth for the validation suite (``tests/test_validation.py``).
2. A microsecond-cost baseline predictor that the learned residual model
   corrects (:mod:`cueai.ml`), which is what makes the "physics-informed"
   framing measurable rather than decorative.

Derivation (ball of mass m, radius R, I = 2/5 mR², struck along +x)
-------------------------------------------------------------------
While the contact point slips, cloth friction acts opposite the slip velocity
u with magnitude μ_s g, and the resulting torque drives u to zero at

    |du/dt| = μ_s g (1 + mR²/I) = 3.5 μ_s g

A tip offset of f·R above centre launches the ball with ω = 2.5 f v₀ / R, so
u₀ = v₀ |1 - 2.5 f|:

    t_slide  = u₀ / (3.5 μ_s g)
    v_roll   = v₀ - s μ_s g t_slide          (s = sign(1 - 2.5f))
    d_slide  = v₀ t_slide - s ½ μ_s g t_slide²
    d_roll   = v_roll² / (2 μ_r g)

For a centre-ball hit (f = 0) this collapses to the familiar
v_roll = 5/7 v₀ and d_slide = 12 v₀² / (49 μ_s g).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cueai.physics.constants import G, ShotParams, TableParams

# Slip decays 3.5x faster than the centre of mass because the friction torque
# also spins the ball up: 1 + mR²/I = 1 + 5/2.
SLIP_DECAY_FACTOR = 3.5
# Fraction of initial speed remaining once a centre-ball hit starts rolling.
ROLLING_SPEED_RATIO = 5.0 / 7.0


@dataclass(frozen=True)
class StraightShot:
    """Phase-resolved solution for a straight shot on an empty table."""

    v_roll: float       # speed at the sliding → rolling transition (m/s)
    t_slide: float      # duration of the sliding phase (s)
    d_slide: float      # distance covered while sliding (m)
    d_roll: float       # distance covered while rolling (m)

    @property
    def d_total(self) -> float:
        return self.d_slide + self.d_roll


def straight_shot(v0: float, table: TableParams, english_y: float = 0.0) -> StraightShot:
    """Closed-form slide/roll decomposition for a straight shot."""
    if v0 <= 0:
        return StraightShot(0.0, 0.0, 0.0, 0.0)

    slip0 = v0 * abs(1.0 - 2.5 * english_y)
    sign = float(np.sign(1.0 - 2.5 * english_y))
    t_slide = slip0 / (SLIP_DECAY_FACTOR * table.mu_slide * G)
    v_roll = v0 - sign * table.mu_slide * G * t_slide
    d_slide = v0 * t_slide - sign * 0.5 * table.mu_slide * G * t_slide**2
    d_roll = v_roll**2 / (2.0 * table.mu_roll * G)
    return StraightShot(
        v_roll=float(v_roll),
        t_slide=float(t_slide),
        d_slide=float(d_slide),
        d_roll=float(d_roll),
    )


def slide_distance(v0: float, mu_slide: float) -> float:
    """Distance to the rolling transition for a centre-ball hit: 12v₀²/(49μg)."""
    return 12.0 * v0**2 / (49.0 * mu_slide * G)


def roll_distance(v_roll: float, mu_roll: float) -> float:
    """Distance a rolling ball covers before stopping: v²/(2μ_r g)."""
    return v_roll**2 / (2.0 * mu_roll * G)


def _reflect(coord: float, lo: float, hi: float) -> float:
    """Fold a coordinate into [lo, hi] as an ideal mirror would."""
    span = hi - lo
    if span <= 0:
        return lo
    folded = (coord - lo) % (2.0 * span)
    if folded > span:
        folded = 2.0 * span - folded
    return lo + folded


def predict_endpoint(
    shot: ShotParams,
    cue_pos: np.ndarray | tuple[float, float],
    table: TableParams,
    radius: float = 0.028575,
) -> np.ndarray:
    """
    Baseline resting position of the cue ball, in closed form.

    Travels the analytic stopping distance along the aim line and mirrors off
    the cushions. Deliberately ignores cushion energy loss, ball-ball contact,
    throw and pockets — those are the corrections the learned model supplies.
    """
    solution = straight_shot(shot.speed, table, english_y=shot.english_y)
    start = np.asarray(cue_pos, dtype=np.float64)
    end = start + solution.d_total * np.array(
        [np.cos(shot.angle), np.sin(shot.angle)], dtype=np.float64
    )
    return np.array(
        [
            _reflect(end[0], radius, table.length - radius),
            _reflect(end[1], radius, table.width - radius),
        ],
        dtype=np.float64,
    )
