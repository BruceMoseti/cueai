"""Rigid-body ball state and cloth dynamics (sliding → rolling → rest)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from cueai.physics.constants import BallParams, G, TableParams

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
    identity: BallIdentity | None = None

    def copy(self) -> Ball:
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
        """
        Velocity of the cloth contact point: u = v + ω × (-Rẑ).

        Expanding the cross product gives u = (vₓ - Rω_y, v_y + Rω_x), so pure
        rolling (u = 0) corresponds to ω_y = vₓ/R and ω_x = -v_y/R.
        """
        return np.array(
            [
                self.vel[0] - self.omega[1] * self.R,
                self.vel[1] + self.omega[0] * self.R,
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
        # Tolerance must exceed the per-step slip decrement (≈3.5 μ g dt) to avoid
        # chattering between SLIDING and ROLLING near the transition.
        slip_eps = max(0.01, 0.01 * v_mag)
        if v_mag < eps and u_mag < eps and wz < eps:
            return MotionState.STATIONARY
        if u_mag < slip_eps and v_mag >= eps:
            return MotionState.ROLLING
        if v_mag < eps and wz >= eps:
            return MotionState.SPINNING
        return MotionState.SLIDING


def decay_spin(omega_z: float, table: TableParams, radius: float, dt: float) -> float:
    """
    Spin-down about the vertical axis, clamped at zero.

    The decrement per step is constant, so subtracting it unconditionally steps
    straight past zero and flips the sign; the ball then chatters between two
    small values and never reaches rest, and `Simulator.run` gives up at
    `max_time` instead of stopping when the table is still. Clamping is not a
    tolerance hack: friction removes spin, it cannot reverse it.
    """
    step = 2.5 * table.mu_spin * G / radius * dt
    if abs(omega_z) <= step:
        return 0.0
    return omega_z - math.copysign(step, omega_z)


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

    if state is MotionState.STATIONARY:
        b.vel[:] = 0
        b.omega[:] = 0
        return b

    if state is MotionState.SPINNING:
        b.omega[2] = decay_spin(float(b.omega[2]), table, R, dt)
        return b

    if state is MotionState.SLIDING:
        u = b.slip_velocity()
        u_mag = float(np.linalg.norm(u))
        if u_mag < 1e-9:
            return b
        u_hat = u / u_mag
        # Friction opposes the contact-point slip, not the ball centre velocity
        a = -mu_s * G * u_hat
        # τ = (-Rẑ) × F  ⇒  α = (R m a_y / I, -R m a_x / I, 0)
        alpha = np.array([R * m * a[1] / I, -R * m * a[0] / I, 0.0])

        b.vel = b.vel + a * dt
        b.omega = b.omega + alpha * dt
        b.omega[2] = decay_spin(float(b.omega[2]), table, R, dt)
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
    # Rolling resistance only. Vertical-axis spin exerts no lateral force on a
    # rolling rigid sphere; it acts through cushion and ball-ball throw instead.
    a = -mu_r * G * v_hat
    b.vel = b.vel + a * dt
    if b.speed() < 2e-2:
        b.vel[:] = 0
        b.omega[:] = 0
        return b
    b.omega[0] = -b.vel[1] / R
    b.omega[1] = b.vel[0] / R
    b.omega[2] = decay_spin(float(b.omega[2]), table, R, dt)
    b.pos = b.pos + b.vel * dt
    return b
