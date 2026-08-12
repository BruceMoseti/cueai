"""Physics constants and table / ball parameters (SI units, meters)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


G = 9.81  # m/s^2


@dataclass(frozen=True)
class BallParams:
    """Standard pool ball (≈2¼ inch)."""

    radius: float = 0.028575  # m
    mass: float = 0.170       # kg

    @property
    def inertia(self) -> float:
        """Solid-sphere moment of inertia I = (2/5) m R²."""
        return 0.4 * self.mass * self.radius**2


@dataclass
class TableParams:
    """American pool table playing surface (9-ft) with friction / restitution."""

    length: float = 2.54   # m (playing area)
    width: float = 1.27    # m
    mu_slide: float = 0.2  # sliding friction cloth
    mu_roll: float = 0.01  # rolling resistance
    mu_spin: float = 0.044 # spinning friction (Alciatore / pooltool-style)
    e_ball: float = 0.95   # ball-ball restitution
    e_cushion: float = 0.85
    mu_ball: float = 0.06  # ball-ball tangential friction (throw)
    mu_cushion: float = 0.20
    # Spatially varying cloth noise (table imperfections)
    friction_noise_amp: float = 0.02
    friction_noise_scale: float = 0.35  # correlation length (m)
    pocket_radius: float = 0.06
    pockets: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.pockets:
            L, W = self.length, self.width
            object.__setattr__(
                self,
                "pockets",
                [
                    (0.0, 0.0),
                    (L / 2, 0.0),
                    (L, 0.0),
                    (0.0, W),
                    (L / 2, W),
                    (L, W),
                ],
            )


@dataclass
class ShotParams:
    """Cue strike parameterization."""

    speed: float          # tip speed imparted → cue-ball linear speed (m/s)
    angle: float          # launch angle radians, 0 = +x
    english_x: float = 0  # sidespin tip offset as fraction of R (+ = right english)
    english_y: float = 0  # topspin (+) / backspin (−) tip offset (-1..1)
    cue_elevation: float = 0.0  # radians (jump / massé lite)

    def initial_omega(self, ball: BallParams) -> np.ndarray:
        """
        Map cue tip offset to initial angular velocity.

        A horizontal impulse J applied a distance d = f·R off centre gives
        Δv = J/m and Δω = J·d/I, so with I = (2/5)mR²:

            ω = 2.5 · f · v / R

        Hence f = 0.4 (tip 2R/5 above centre, i.e. 7R/5 above the cloth)
        launches the ball already rolling — the standard "natural roll" result —
        and |f| > 0.5 is past the practical miscue limit.

        - Top/backspin acts about the horizontal axis perpendicular to travel,
          which for a shot along θ is (-sin θ, cos θ, 0) — the same axis as
          natural roll, so english_y = +1 is pure follow.
        - Sidespin acts about the vertical axis; positive english_x (right
          english) is clockwise seen from above, hence negative ω_z.
        """
        R = ball.radius
        v = self.speed
        roll_axis = np.array([-np.sin(self.angle), np.cos(self.angle), 0.0])
        omega = 2.5 * self.english_y * (v / R) * roll_axis
        omega[2] -= 2.5 * self.english_x * v / R
        return omega
