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
    english_x: float = 0  # sidespin tip offset as fraction of R (-1..1)
    english_y: float = 0  # topspin / backspin tip offset (-1..1)
    cue_elevation: float = 0.0  # radians (jump / massé lite)

    def initial_omega(self, ball: BallParams) -> np.ndarray:
        """Map tip offset to angular velocity (simplified cue model)."""
        R = ball.radius
        # Tip contact ≈ tangential impulse → ω ≈ (v_tip × r) / (k R)
        # Use Alciatore-style: ω_y ~ -english_x * v / R, ω_x ~ english_y * v / R
        v = self.speed
        wx = self.english_y * v / R
        wy = -self.english_x * v / R
        wz = 0.15 * self.english_x * v / R  # slight vertical-axis from sidespin
        return np.array([wx, wy, wz], dtype=np.float64)
