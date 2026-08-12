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

from pocket.physics import analytic
from pocket.physics.ball import Ball, MotionState, integrate_ball
from pocket.physics.collisions import (
    CONTACT_BAND,
    resolve_all_ball_collisions,
    resolve_ball_ball,
    resolve_cushion,
)
from pocket.physics.constants import BallParams, G, ShotParams, TableParams
from pocket.physics.rack import make_full_rack
from pocket.physics.simulator import Simulator

DT = 1e-4
SMOOTH_CLOTH = {"friction_noise_amp": 0.0}


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


@pytest.mark.parametrize(
    "label,angle_deg,obj_pos,tolerance_m",
    [
        ("full ball", 0.0, (1.4, 0.635), 0.01),
        # A thin cut amplifies the contact geometry, so it is the worst case for
        # the timestep and gets the looser bound.
        ("half ball", 2.0, (1.5, 0.66), 0.06),
    ],
)
def test_training_timestep_is_converged(
    label: str, angle_deg: float, obj_pos: tuple[float, float], tolerance_m: float
) -> None:
    """
    Labels are generated at 2 ms, so halving the step must barely move the answer.

    Without this bound the reported model errors could be measuring the
    integrator's discretisation rather than the model, and no amount of training
    would fix it. See docs/VALIDATION.md for the measured values.
    """
    table = TableParams(**SMOOTH_CLOTH)
    shot = ShotParams(speed=3.0 if angle_deg else 2.5, angle=float(np.radians(angle_deg)))
    cue_pos = (0.6, 0.60) if angle_deg else (0.6, 0.635)

    endpoints = {}
    for dt in (2e-3, 1e-3):
        result = Simulator(table=table, dt=dt, max_time=8.0).simulate_shot(
            shot, cue_pos=cue_pos, obj_pos=obj_pos
        )
        endpoints[dt] = (result.endpoints[0].copy(), result.endpoints[1].copy())
        assert result.collision_events > 0, "the shot must actually make contact"

    for index, ball in enumerate(("cue", "object")):
        gap = float(np.linalg.norm(endpoints[2e-3][index] - endpoints[1e-3][index]))
        assert gap < tolerance_m, f"{label} {ball} ball moved {gap * 1000:.0f} mm"


def test_ghost_ball_geometry_pots_and_half_a_degree_misses() -> None:
    """
    Contact geometry against its closed form, and the tolerance that follows.

    Aiming at the ghost-ball point — the object ball's centre pulled one diameter
    back along the line to the pocket — must pot the ball. Half a degree off must
    not, which is the measurement behind the claim that potting needs finer aim
    than the surrogate can resolve.
    """
    table = TableParams(**SMOOTH_CLOTH)
    sim = Simulator(table=table, dt=2e-3, max_time=8.0)
    cue_pos, obj_pos = (0.60, 0.35), (1.70, 0.75)
    pocket = np.array([table.length, table.width])

    obj = np.asarray(obj_pos)
    to_pocket = (pocket - obj) / np.linalg.norm(pocket - obj)
    ghost = obj - 2 * sim.ball_params.radius * to_pocket
    aim = float(np.arctan2(*(ghost - np.asarray(cue_pos))[::-1]))

    def pots(angle: float) -> bool:
        result = sim.simulate_shot(
            ShotParams(speed=2.5, angle=angle), cue_pos=cue_pos, obj_pos=obj_pos
        )
        return bool(result.pocketed[1]) and not result.pocketed[0]

    assert pots(aim), "the closed-form ghost-ball line must pot the object ball"
    for offset_deg in (-0.5, 0.5):
        assert not pots(aim + float(np.radians(offset_deg))), (
            f"{offset_deg:+.1f}° off the ghost-ball line should miss"
        )


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


def test_vertical_spin_decays_to_zero_without_chattering() -> None:
    """
    Spinning friction must land on zero, not step past it.

    The decrement per timestep is constant, so subtracting it unconditionally
    overshoots and flips the sign, leaving the ball spinning at a small
    alternating rate for ever. Nothing downstream notices except that the table
    never comes to rest, which is why this is asserted directly.
    """
    table = TableParams(**SMOOTH_CLOTH)
    params = BallParams()
    dt = 1e-3
    # Deliberately start below one step's worth of decay, where overshoot bites.
    step = 2.5 * table.mu_spin * G / params.radius * dt
    ball = Ball(
        id=0,
        pos=np.array([1.0, 0.635]),
        vel=np.zeros(2),
        omega=np.array([0.0, 0.0, 0.4 * step]),
        params=params,
    )

    previous = abs(float(ball.omega[2]))
    for _ in range(50):
        ball = integrate_ball(ball, table, dt)
        current = abs(float(ball.omega[2]))
        assert current <= previous + 1e-12, "spin grew while friction was removing it"
        previous = current

    assert float(ball.omega[2]) == 0.0
    assert ball.motion_state(table) is MotionState.STATIONARY


@pytest.mark.parametrize(
    ("label", "shot"),
    [
        ("soft into the rack", ShotParams(speed=2.5, angle=0.12, english_x=0.2, english_y=-0.1)),
        ("hard break", ShotParams(speed=8.0, angle=0.0)),
        ("heavy right english", ShotParams(speed=4.0, angle=0.35, english_x=0.48)),
        ("draw into the pack", ShotParams(speed=5.0, angle=-0.05, english_y=-0.45)),
    ],
)
def test_every_shot_reaches_rest_before_the_time_limit(label: str, shot: ShotParams) -> None:
    """
    A shot must end because the balls stopped, not because the clock ran out.

    Hitting ``max_time`` silently truncates the shot, and a truncated shot is
    still reported as a resting position, so the failure is invisible in the
    output and shows up only as a simulator that will not terminate.
    """
    sim = Simulator(dt=1e-3, max_time=15.0)
    result = sim.simulate_shot(shot, full_rack=True, seed=7)
    settled_at = float(result.times[-1])
    assert settled_at < sim.max_time - 1.0, (
        f"{label}: still moving at {settled_at:.1f}s, the {sim.max_time:.0f}s limit truncated it"
    )


def test_every_contact_in_a_rack_is_inside_the_contact_band() -> None:
    """
    A racked ball touches its neighbours, and the solver has to agree.

    This is the regression test for a bug that was invisible in every other
    check. The contact band used to be ``1e-4`` and the rack was built with a
    ``1e-4`` clearance, so all thirty contacts in the triangle sat exactly on
    the threshold that decides whether a contact exists. Which side of it each
    one landed on came down to whether ``hypot`` rounded up or down: sixteen
    registered, fourteen did not. A break then propagated through a contact
    graph with holes in it, and balls in the middle of the rack came out of a
    full-power break having never moved.

    Neither the closed-form checks nor the parity harness could see it: the
    single-ball mechanics were untouched, and the JavaScript port reproduced
    the broken graph exactly, because it was a faithful port of it.
    """
    balls = make_full_rack(TableParams())[1:]
    touching = 0
    for i, a in enumerate(balls):
        for b in balls[i + 1 :]:
            gap = float(np.hypot(*(b.pos - a.pos))) - a.R - b.R
            if gap > 1e-3:
                continue  # not neighbours in the triangle
            touching += 1
            assert gap <= CONTACT_BAND, (
                f"balls {a.number} and {b.number} are {gap * 1e6:.3f} µm apart, "
                f"outside the {CONTACT_BAND * 1e6:.0f} µm contact band, so the "
                "solver will not see them as touching"
            )
    assert touching == 30, f"a five-row triangle has 30 contacts, found {touching}"


def test_the_contact_band_is_far_from_both_things_that_could_swallow_it() -> None:
    """
    The band has to be wide against float noise and narrow against physics.

    Stated as a test because the failure mode is silent either way: too tight
    and real contacts are missed, too loose and balls collide with thin air.
    """
    noise = 1e-15  # the scale of rounding in a position, in metres
    assert CONTACT_BAND / noise > 1000
    assert BallParams().radius / CONTACT_BAND > 1000


def test_resolving_a_resting_rack_changes_nothing_and_stops_immediately() -> None:
    """
    Thirty resting contacts must not cost thirty passes of work every step.

    The sweep exits when a pass changes nothing, so a table that is merely
    sitting there costs one pass. Without that, making the rack's contacts
    visible would have quietly multiplied the cost of every timestep.
    """
    table = TableParams()
    before = make_full_rack(table)
    after = resolve_all_ball_collisions(before, table)
    for a, b in zip(before, after):
        assert np.allclose(a.pos, b.pos), f"ball {a.number} was moved while at rest"
        assert np.allclose(a.vel, b.vel), f"ball {a.number} was pushed while at rest"


def test_a_break_puts_every_ball_in_the_rack_in_motion() -> None:
    """
    The impulse has to reach the whole triangle, not the near half of it.

    Two deliberately loose claims, because a break is chaotic and a tight
    assertion about it is a lottery: every ball is set moving, and most of the
    rack ends up somewhere else. What the numbers are is not the point; that
    there is no ball the impulse never reached is.

    The distribution across those balls is much less even than a real break —
    see the multi-contact note in ``docs/VALIDATION.md``. Resolving contacts
    pairwise in sequence is an approximation to a rack in which fifteen balls
    are touching at once, and it is the largest known departure from reality in
    this simulator.
    """
    sim = Simulator(dt=1e-3, max_time=20.0)
    result = sim.simulate_shot(ShotParams(speed=8.0, angle=0.0), full_rack=True, seed=7)

    untouched, displaced = [], 0
    for ball_id, path in result.trajectories.items():
        if ball_id == 0:
            continue
        if float(np.max(np.linalg.norm(result.velocities[ball_id], axis=1))) < 1e-3:
            untouched.append(ball_id)
        displaced += float(np.linalg.norm(path[-1] - path[0])) > 0.05

    assert not untouched, f"balls {sorted(untouched)} never moved on a full-power break"
    assert displaced >= 8, f"only {displaced} of 15 balls left the rack area"


def test_a_harder_break_opens_the_table_further() -> None:
    """
    The property whose sign was wrong, asserted so it cannot go wrong quietly.

    Nobody writes this test first, because it is too obvious to be worth
    stating. That is exactly why the contact-band defect survived: every
    closed-form check passed, the browser port agreed to eleven decimal places,
    and the only visible symptom was that a rack struck at 8 m/s ended up
    *tighter* than one struck at 5. The measurement that found it is the one
    below.

    Coarse on purpose. A 3 ms step and one rack per speed keeps this to a few
    seconds, and the claim is about the direction of an aggregate, which is not
    a quantity a millimetre of discretisation decides.
    """
    spreads = []
    for speed in (3.0, 6.0, 9.0):
        sim = Simulator(dt=3e-3, max_time=15.0)
        result = sim.simulate_shot(ShotParams(speed=speed, angle=0.0), full_rack=True, seed=7)
        resting = np.array(
            [
                result.endpoints[ball_id][:2]
                for ball_id in result.trajectories
                if ball_id != 0 and not result.pocketed[ball_id]
            ]
        )
        centre = resting.mean(axis=0)
        spreads.append(float(np.mean(np.linalg.norm(resting - centre, axis=1))))

    assert spreads[0] < spreads[1] < spreads[2], (
        "a harder break has to leave the balls further apart, but the mean "
        f"distance from the centre of the pack went {spreads[0]:.3f} → "
        f"{spreads[1]:.3f} → {spreads[2]:.3f} m at 3, 6 and 9 m/s"
    )
