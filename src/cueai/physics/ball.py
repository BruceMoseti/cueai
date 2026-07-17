"""Rigid-body ball state and cloth dynamics (sliding → rolling → rest)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import numpy as np

from cueai.physics.constants import G, BallParams, TableParams

if TYPE_CHECKING:
    from cueai.physics.rack import BallIdentity


class MotionState(Enum):
    SLIDING = auto()
    ROLLING = auto()
    SPINNING = auto()
    STATIONARY = auto()
    POCKETED = auto()


@dataclass
class Ball:
    id: int
    pos: np.ndarray          # (x, y) meters
    vel: np.ndarray          # (vx, vy)
    omega: np.ndarray        # (ωx, ωy, ωz) rad/s
    params: BallParams = field(default_factory=BallParams)
    pocketed: bool = False
    number: int = 0
    identity: Optional["BallIdentity"] = None

    def copy(self) -> "Ball":
        return Ball(
            id=self.id,
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            omega=self.omega.copy(),
            params=self.params,
            pocketed=self.pocketed,
            number=self.number,
            identity=self.identity,
        )

    @property
    def R(self) -> float:
        return self.params.radius

    @property
    def m(self) -> float:
        return self.params.mass

    @property
    def I(self) -> float:
        return self.params.inertia

    def slip_velocity(self) -> np.ndarray:
        """Velocity of cloth contact point: u = v + ω × (-Rẑ)."""
        return np.array(
            [
                self.vel[0] + self.omega[1] * self.R,
                self.vel[1] - self.omega[0] * self.R,
            ],
            dtype=np.float64,
        )

    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    def motion_state(self, table: TableParams, eps: float = 1e-3) -> MotionState:
        if self.pocketed:
            return MotionState.POCKETED
        u = self.slip_velocity()
        u_mag = float(np.linalg.norm(u))
        v_mag = self.speed()
        wz = abs(self.omega[2])
        slip_eps = max(0.05, 0.05 * max(v_mag, 0.1))
        if v_mag < eps and u_mag < eps and wz < eps:
            return MotionState.STATIONARY
        if u_mag < slip_eps and v_mag >= eps:
            return MotionState.ROLLING
        if v_mag < eps and wz >= eps:
            return MotionState.SPINNING
        return MotionState.SLIDING


def local_mu_slide(table: TableParams, pos: np.ndarray) -> float:
    """Spatially varying sliding friction (table imperfections via smooth noise)."""
    amp = table.friction_noise_amp
    if amp <= 0:
        return table.mu_slide
    scale = table.friction_noise_scale
    nx = np.sin(pos[0] / scale * 2 * np.pi) * np.cos(pos[1] / scale * 2 * np.pi)
    ny = np.sin((pos[0] + pos[1]) / scale * np.pi)
    return float(np.clip(table.mu_slide + amp * 0.5 * (nx + ny), 0.05, 0.45))


def integrate_ball(ball: Ball, table: TableParams, dt: float) -> Ball:
    """One Euler step of cloth dynamics (Kiefl / Marlow / Alciatore style)."""
    if ball.pocketed:
        return ball

    b = ball.copy()
    state = b.motion_state(table)
    R, m, I = b.R, b.m, b.I
    mu_s = local_mu_slide(table, b.pos)
    mu_r = table.mu_roll
    mu_sp = table.mu_spin

    if state is MotionState.STATIONARY:
        b.vel[:] = 0
        b.omega[:] = 0
        return b

    if state is MotionState.SPINNING:
        decay = 2.5 * mu_sp * G / R
        sgn = np.sign(b.omega[2]) if b.omega[2] != 0 else 0.0
        b.omega[2] -= decay * sgn * dt
        if abs(b.omega[2]) < 1e-4:
            b.omega[2] = 0.0
        return b

    if state is MotionState.SLIDING:
        u = b.slip_velocity()
        u_mag = float(np.linalg.norm(u))
        if u_mag < 1e-9:
            return b
        u_hat = u / u_mag
        # Curving force: slip has a component from sidespin → path curves
        a = -mu_s * G * u_hat
        alpha = np.array([-R * m * a[1] / I, R * m * a[0] / I, 0.0])
        if abs(b.omega[2]) > 1e-6:
            alpha[2] = -np.sign(b.omega[2]) * 2.5 * mu_sp * G / R

        b.vel = b.vel + a * dt
        b.omega = b.omega + alpha * dt
        u2 = b.slip_velocity()
        if float(np.linalg.norm(u2)) < max(1e-3, 0.01 * b.speed()):
            b.omega[0] = -b.vel[1] / R
            b.omega[1] = b.vel[0] / R
        b.pos = b.pos + b.vel * dt
        return b

    v_mag = b.speed()
    if v_mag < 1e-9:
        b.vel[:] = 0
        b.omega[:2] = 0
        return b
    v_hat = b.vel / v_mag
    # Mild curve while rolling if residual ωz (english hold)
    a = -mu_r * G * v_hat
    if abs(b.omega[2]) > 0.5:
        # small lateral force from residual spin-on-cloth coupling
        lateral = np.array([-v_hat[1], v_hat[0]]) * (0.002 * b.omega[2])
        a = a + lateral
    b.vel = b.vel + a * dt
    if b.speed() < 2e-2:
        b.vel[:] = 0
        b.omega[:] = 0
        return b
    b.omega[0] = -b.vel[1] / R
    b.omega[1] = b.vel[0] / R
    if abs(b.omega[2]) > 1e-6:
        b.omega[2] -= np.sign(b.omega[2]) * 2.5 * mu_sp * G / R * dt
    b.pos = b.pos + b.vel * dt
    return b
