"""
Validation of the numerical simulator against closed-form physics.

Every test here compares the integrator to an independently derived analytic
result or to a conservation law, so a failure means the physics is wrong rather
than merely different. The tolerances are the numbers quoted in
``docs/VALIDATION.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cueai.physics import analytic
from cueai.physics.ball import Ball, MotionState, integrate_ball
from cueai.physics.collisions import resolve_ball_ball, resolve_cushion
from cueai.physics.constants import G, BallParams, ShotParams, TableParams
from cueai.physics.simulator import Simulator

DT = 1e-4
SMOOTH_CLOTH = dict(friction_noise_amp=0.0)


def _launch(v0: float, english_y: float = 0.0, english_x: float = 0.0) -> Ball:
    params = BallParams()
    shot = ShotParams(speed=v0, angle=0.0, english_x=english_x, english_y=english_y)
    return Ball(
        id=0,
        pos=np.array([0.0, 0.635]),
        vel=np.array([v0, 0.0]),
        omega=shot.initial_omega(params),
        params=params,
    )


def _roll_out(
    ball: Ball,
    table: TableParams,
    dt: float = DT,
    max_time: float = 60.0,
    stop_at_transition: bool = False,
):
    """Integrate an unobstructed ball to rest, capturing the rolling transition."""
    transition: tuple[float, float] | None = None
    for _ in range(int(max_time / dt)):
        ball = integrate_ball(ball, table, dt)
        if transition is None and ball.motion_state(table) is MotionState.ROLLING:
            transition = (ball.speed(), float(ball.pos[0]))
            if stop_at_transition:
                break
        if ball.speed() == 0.0:
            break
    return ball, transition


@pytest.mark.parametrize("v0", [1.0, 2.0, 4.0])
def test_rolling_speed_is_five_sevenths_of_launch_speed(v0: float) -> None:
    """Classic result: a centre-ball hit starts rolling at 5/7 v₀."""
    table = TableParams(**SMOOTH_CLOTH)
    _, transition = _roll_out(_launch(v0), table, stop_at_transition=True)
    assert transition is not None
    v_roll, _ = transition
    assert v_roll == pytest.approx(analytic.ROLLING_SPEED_RATIO * v0, rel=0.01)


@pytest.mark.parametrize("v0", [1.0, 2.0, 4.0])
def test_slide_distance_matches_closed_form(v0: float) -> None:
    """Sliding phase length must equal 12v₀²/(49 μ_s g)."""
    table = TableParams(**SMOOTH_CLOTH)
    _, transition = _roll_out(_launch(v0), table, stop_at_transition=True)
    assert transition is not None
    _, x_roll = transition
    assert x_roll == pytest.approx(analytic.slide_distance(v0, table.mu_slide), rel=0.02)


@pytest.mark.parametrize("v0", [1.0, 2.0, 3.0])
def test_total_stopping_distance_matches_closed_form(v0: float) -> None:
    """Slide + roll distance on an unbounded table, within 1%."""
    # A brisk cloth (mu_roll=0.03) keeps the integration short enough for CI.
    table = TableParams(mu_roll=0.03, **SMOOTH_CLOTH)
    ball, _ = _roll_out(_launch(v0), table)
    expected = analytic.straight_shot(v0, table).d_total
    assert float(ball.pos[0]) == pytest.approx(expected, rel=0.01)


def test_natural_roll_tip_offset_produces_no_sliding_phase() -> None:
    """A tip 2R/5 above centre launches the ball already rolling."""
    table = TableParams(mu_roll=0.03, **SMOOTH_CLOTH)
    ball = _launch(2.0, english_y=0.4)
    assert ball.motion_state(table) is MotionState.ROLLING
    assert float(np.linalg.norm(ball.slip_velocity())) < 1e-9
    stopped, _ = _roll_out(ball, table)
    expected = analytic.roll_distance(2.0, table.mu_roll)
    assert float(stopped.pos[0]) == pytest.approx(expected, rel=0.01)
    assert float(stopped.pos[1]) == pytest.approx(0.635, abs=1e-9)


def test_spin_decays_at_analytic_rate() -> None:
    """Vertical-axis spin decays linearly at 5 μ_sp g / 2R."""
    table = TableParams(**SMOOTH_CLOTH)
    params = BallParams()
    omega0 = 60.0
    ball = Ball(
        id=0,
        pos=np.array([1.0, 0.6]),
        vel=np.zeros(2),
        omega=np.array([0.0, 0.0, omega0]),
        params=params,
    )
    steps = 500
    for _ in range(steps):
        ball = integrate_ball(ball, table, DT)
    expected = omega0 - 2.5 * table.mu_spin * G / params.radius * steps * DT
    assert float(ball.omega[2]) == pytest.approx(expected, rel=1e-6)


def test_sidespin_does_not_deflect_a_rolling_ball() -> None:
    """A rolling rigid sphere feels no lateral force from vertical-axis spin."""
    table = TableParams(mu_roll=0.03, **SMOOTH_CLOTH)
    ball = _launch(2.0, english_y=0.4, english_x=0.5)
    assert abs(float(ball.omega[2])) > 1.0
    stopped, _ = _roll_out(ball, table)
    assert float(stopped.pos[1]) == pytest.approx(0.635, abs=1e-9)


def test_ball_ball_collision_conserves_linear_momentum() -> None:
    """Frictional collisions must conserve momentum to machine precision."""
    params = BallParams()
    table = TableParams()
    a = Ball(
        id=0,
        pos=np.array([0.0, 0.0]),
        vel=np.array([3.0, 0.7]),
        omega=np.array([5.0, 60.0, -30.0]),
        params=params,
    )
    b = Ball(
        id=1,
        pos=np.array([2 * params.radius, 0.0]),
        vel=np.zeros(2),
        omega=np.zeros(3),
        params=params,
    )
    before = params.mass * (a.vel + b.vel)
    a_out, b_out = resolve_ball_ball(a, b, table)
    after = params.mass * (a_out.vel + b_out.vel)
    np.testing.assert_allclose(after, before, atol=1e-12)


def _kinetic_energy(balls: list[Ball]) -> float:
    return float(
        sum(
            0.5 * b.m * b.vel @ b.vel + 0.5 * b.I * b.omega @ b.omega
            for b in balls
        )
    )


def test_collisions_never_create_energy() -> None:
    """Inelastic contacts are strictly dissipative, for ball-ball and cushions."""
    params = BallParams()
    table = TableParams()
    a = Ball(
        id=0,
        pos=np.array([1.0, 0.6]),
        vel=np.array([4.0, -1.2]),
        omega=np.array([10.0, 80.0, 40.0]),
        params=params,
    )
    b = Ball(
        id=1,
        pos=np.array([1.0 + 2 * params.radius, 0.6]),
        vel=np.array([0.2, 0.0]),
        omega=np.zeros(3),
        params=params,
    )
    before = _kinetic_energy([a, b])
    a_out, b_out = resolve_ball_ball(a, b, table)
    assert _kinetic_energy([a_out, b_out]) <= before + 1e-12

    rail = Ball(
        id=2,
        pos=np.array([params.radius - 1e-4, 0.6]),
        vel=np.array([-3.0, 0.5]),
        omega=np.array([0.0, 0.0, 25.0]),
        params=params,
    )
    before_rail = _kinetic_energy([rail])
    assert _kinetic_energy([resolve_cushion(rail, table)]) <= before_rail + 1e-12


@pytest.mark.parametrize("speed", [1.0, 3.0])
def test_cushion_restitution_matches_configured_coefficient(speed: float) -> None:
    """Head-on rail rebound speed equals e_cushion x approach speed."""
    params = BallParams()
    table = TableParams(e_cushion=0.85)
    ball = Ball(
        id=0,
        pos=np.array([params.radius - 1e-5, 0.635]),
        vel=np.array([-speed, 0.0]),
        omega=np.zeros(3),
        params=params,
    )
    out = resolve_cushion(ball, table)
    # Restitution is softened above 1.5 m/s to mimic cushion compliance.
    expected = min(table.e_cushion, max(0.55, table.e_cushion - 0.03 * max(0.0, speed - 1.5)))
    assert float(out.vel[0]) == pytest.approx(expected * speed, rel=1e-9)


def test_draw_stun_follow_are_correctly_ordered() -> None:
    """
    Cue ball behaviour after a full-ball contact must follow the known ordering:
    backspin draws it back behind contact, topspin sends it furthest forward.
    """
    sim = Simulator(dt=2e-4, max_time=12.0, collision_passes=10)
    sim.table = TableParams(**SMOOTH_CLOTH)
    contact_x = 1.2 - 2 * sim.ball_params.radius

    endpoints = {}
    for label, english_y in (("draw", -0.45), ("stun", 0.0), ("follow", 0.45)):
        result = sim.simulate_shot(
            ShotParams(speed=2.5, angle=0.0, english_y=english_y),
            cue_pos=(0.6, 0.635),
            obj_pos=(1.2, 0.635),
            full_rack=False,
        )
        endpoints[label] = float(result.endpoints[0][0])

    assert endpoints["draw"] < contact_x, "backspin must pull the cue ball back"
    assert endpoints["draw"] < endpoints["stun"] < endpoints["follow"]


def test_analytic_baseline_tracks_simulator_on_open_table() -> None:
    """The closed-form baseline agrees with the simulator when no rail is hit."""
    table = TableParams(mu_slide=0.2, mu_roll=0.05, **SMOOTH_CLOTH)
    sim = Simulator(table=table, dt=5e-4, max_time=20.0)
    shot = ShotParams(speed=1.2, angle=0.0)
    result = sim.simulate_shot(shot, cue_pos=(0.3, 0.635), object_ball=False)
    predicted = analytic.predict_endpoint(shot, (0.3, 0.635), table)
    assert float(result.endpoints[0][0]) == pytest.approx(float(predicted[0]), abs=0.02)


def test_analytic_baseline_tracks_simulator_across_a_cushion() -> None:
    """
    Agreement must survive a rail, which is only true if the closed-form model
    carries spin through the bounce: a ball arrives at the next rail sliding.
    """
    table = TableParams(mu_slide=0.2, mu_roll=0.02, **SMOOTH_CLOTH)
    sim = Simulator(table=table, dt=5e-4, max_time=30.0)
    shot = ShotParams(speed=2.0, angle=0.0)
    start = (0.4, 0.5)
    result = sim.simulate_shot(shot, cue_pos=start, object_ball=False)
    assert result.cushion_events >= 1
    predicted = analytic.predict_endpoint(shot, start, table)
    error = float(np.linalg.norm(result.endpoints[0] - predicted))
    assert error < 0.10, f"closed-form baseline off by {error * 1000:.0f} mm"
