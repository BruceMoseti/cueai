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

from cueai.physics.constants import G, BallParams, ShotParams, TableParams

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


MAX_EVENTS = 32
# Spatial resolution used to test a segment against the pocket mouths.
POCKET_SCAN_STEP = 0.01


@dataclass(frozen=True)
class FreeBallOutcome:
    """Where an unobstructed ball ends up, and what happened on the way."""

    position: np.ndarray
    potted: bool
    cushions: int


def _slide_rail_time(
    pos: np.ndarray, vel: np.ndarray, accel: np.ndarray, lo: np.ndarray, hi: np.ndarray, t_end: float
) -> tuple[float, int]:
    """
    First cushion crossing during a parabolic sliding segment.

    Solves ½a t² + v t + (p - bound) = 0 per axis and boundary, keeping the
    earliest root inside (0, t_end].
    """
    best_t, best_axis = t_end, -1
    for axis in (0, 1):
        for bound in (lo[axis], hi[axis]):
            a, b, c = 0.5 * accel[axis], vel[axis], pos[axis] - bound
            if abs(a) < 1e-14:
                roots = [-c / b] if abs(b) > 1e-14 else []
            else:
                disc = b * b - 4 * a * c
                if disc < 0:
                    continue
                sqrt_disc = float(np.sqrt(disc))
                roots = [(-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)]
            for root in roots:
                if 1e-9 < root <= best_t:
                    best_t, best_axis = float(root), axis
    return best_t, best_axis


def _ray_rail_distance(
    pos: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[float, int]:
    """Distance to the first cushion along a straight rolling segment."""
    best_distance, best_axis = np.inf, -1
    for axis in (0, 1):
        component = direction[axis]
        if abs(component) < 1e-12:
            continue
        bound = hi[axis] if component > 0 else lo[axis]
        distance = (bound - pos[axis]) / component
        if 1e-9 < distance < best_distance:
            best_distance, best_axis = float(distance), axis
    return best_distance, best_axis


def _first_pocket_hit(points: np.ndarray, table: TableParams) -> int:
    """Index of the first sampled point inside a pocket mouth, or -1."""
    if not table.pockets or len(points) == 0:
        return -1
    mouths = np.asarray(table.pockets, dtype=np.float64)
    distances = np.linalg.norm(points[:, None, :] - mouths[None, :, :], axis=2)
    inside = np.nonzero((distances < table.pocket_radius).any(axis=1))[0]
    return int(inside[0]) if len(inside) else -1


def _sample_count(length: float) -> int:
    return int(np.clip(np.ceil(abs(length) / POCKET_SCAN_STEP), 2, 4096))


def solve_free_ball(
    pos: np.ndarray,
    vel: np.ndarray,
    spin_contact: np.ndarray,
    table: TableParams,
    radius: float = 0.028575,
) -> FreeBallOutcome:
    """
    Closed-form trajectory of a single ball on an otherwise empty table.

    Integrates nothing. The motion is a sequence of exactly solvable segments:

    * **Sliding.** The slip velocity u decays along a fixed direction, so the
      friction force is constant and the path is a parabola of known duration
      |u| / (3.5 μ_s g).
    * **Rolling.** A straight line of length v² / (2 μ_r g).
    * **Cushion.** The normal velocity component reverses and loses energy while
      the spin term u - v carries through, which is what makes a ball arrive at
      the next rail sliding rather than rolling.

    ``spin_contact`` is ω × (-Rẑ), the spin contribution to the contact-point
    velocity, so the slip velocity is ``vel + spin_contact``.

    Not modelled: ball-ball contact, rail friction and spin transfer, the
    speed-dependent softening of the cushions, and cloth inhomogeneity.
    """
    pos = np.asarray(pos, dtype=np.float64).copy()
    vel = np.asarray(vel, dtype=np.float64).copy()
    spin = np.asarray(spin_contact, dtype=np.float64).copy()
    lo = np.array([radius, radius])
    hi = np.array([table.length - radius, table.width - radius])
    cushions = 0

    for _ in range(MAX_EVENTS):
        slip = vel + spin
        slip_speed = float(np.linalg.norm(slip))
        speed = float(np.linalg.norm(vel))

        if slip_speed > 1e-4:
            accel = -table.mu_slide * G * (slip / slip_speed)
            t_slide = slip_speed / (SLIP_DECAY_FACTOR * table.mu_slide * G)
            t_hit, axis = _slide_rail_time(pos, vel, accel, lo, hi, t_slide)
            t = min(t_slide, t_hit) if axis >= 0 else t_slide

            steps = np.linspace(0.0, t, _sample_count(speed * t))[:, None]
            path = pos + vel * steps + 0.5 * accel * steps**2
            hit = _first_pocket_hit(path, table)
            if hit >= 0:
                return FreeBallOutcome(path[hit], True, cushions)

            pos = pos + vel * t + 0.5 * accel * t**2
            vel = vel + accel * t
            spin = (slip - SLIP_DECAY_FACTOR * table.mu_slide * G * (slip / slip_speed) * t) - vel
            if axis < 0 or t_hit >= t_slide:
                spin = -vel  # slip exhausted: the ball is now rolling
                continue
        else:
            if speed < 1e-4:
                return FreeBallOutcome(pos, False, cushions)
            direction = vel / speed
            stop_distance = roll_distance(speed, table.mu_roll)
            rail_distance, axis = _ray_rail_distance(pos, direction, lo, hi)
            travel = min(stop_distance, rail_distance)

            steps = np.linspace(0.0, travel, _sample_count(travel))[:, None]
            path = pos + direction * steps
            hit = _first_pocket_hit(path, table)
            if hit >= 0:
                return FreeBallOutcome(path[hit], True, cushions)

            pos = pos + direction * travel
            if stop_distance <= rail_distance:
                return FreeBallOutcome(pos, False, cushions)
            speed = float(np.sqrt(max(speed**2 - 2 * table.mu_roll * G * travel, 0.0)))
            vel = speed * direction
            spin = -vel

        # Cushion impulse: reverse and damp the normal velocity component. The
        # spin term is untouched, so the ball leaves the rail sliding.
        vel[axis] *= -table.e_cushion
        pos[axis] = float(np.clip(pos[axis], lo[axis], hi[axis]))
        cushions += 1

    return FreeBallOutcome(pos, False, cushions)


def predict_endpoint(
    shot: ShotParams,
    cue_pos: np.ndarray | tuple[float, float],
    table: TableParams,
    radius: float = 0.028575,
) -> np.ndarray:
    """Closed-form resting position of the cue ball for a given shot."""
    omega = shot.initial_omega(BallParams(radius=radius))
    # ω × (-Rẑ) = (-R ω_y, R ω_x)
    spin_contact = np.array([-radius * omega[1], radius * omega[0]])
    velocity = shot.speed * np.array([np.cos(shot.angle), np.sin(shot.angle)])
    return solve_free_ball(
        np.asarray(cue_pos, dtype=np.float64), velocity, spin_contact, table, radius
    ).position
