"""Unit tests for CueAI physics + full rack."""

from __future__ import annotations

import numpy as np

from cueai.physics.ball import Ball, MotionState, integrate_ball
from cueai.physics.constants import BallParams, ShotParams, TableParams
from cueai.physics.rack import make_full_rack
from cueai.physics.simulator import Simulator, shot_feature_vector


def test_ball_comes_to_rest():
    table = TableParams()
    b = Ball(
        id=0,
        pos=np.array([1.0, 0.6]),
        vel=np.array([1.5, 0.0]),
        omega=np.zeros(3),
        params=BallParams(),
    )
    for _ in range(20000):
        b = integrate_ball(b, table, 0.001)
        if b.motion_state(table) is MotionState.STATIONARY:
            break
    assert b.speed() < 1e-2


def test_full_rack_has_sixteen_balls():
    balls = make_full_rack()
    assert len(balls) == 16
    numbers = sorted(b.number for b in balls)
    assert numbers == list(range(16))
    assert any(b.number == 8 for b in balls)


def test_break_spreads_rack():
    sim = Simulator(dt=0.0015, max_time=8.0, collision_passes=10)
    shot = ShotParams(speed=6.0, angle=0.0, english_x=0.15, english_y=-0.1)
    result = sim.simulate_shot(shot, full_rack=True)
    assert len(result.trajectories) == 16
    assert result.collision_events > 5
    # At least a few object balls should have moved from the pack
    moved = 0
    for bid, traj in result.trajectories.items():
        if bid == 0:
            continue
        if len(traj) > 2 and float(np.linalg.norm(traj[-1] - traj[0])) > 0.05:
            moved += 1
    assert moved >= 3


def test_simulate_shot_produces_trajectory():
    sim = Simulator(dt=0.002, max_time=5.0)
    shot = ShotParams(speed=3.0, angle=0.1, english_x=0.2, english_y=-0.1)
    result = sim.simulate_shot(shot, full_rack=False)
    assert 0 in result.trajectories
    assert len(result.trajectories[0]) > 10


def test_feature_vector_dim():
    shot = ShotParams(2.0, 0.5)
    x = shot_feature_vector(shot, np.array([0.5, 0.5]), np.array([1.5, 0.5]), TableParams())
    assert x.shape == (18,)


def test_collision_transfers_momentum():
    sim = Simulator(dt=0.001, max_time=3.0)
    shot = ShotParams(speed=4.0, angle=0.0)
    result = sim.simulate_shot(
        shot,
        cue_pos=(0.8, 0.635),
        obj_pos=(1.2, 0.635),
        full_rack=False,
    )
    # The object ball starts at rest, so any speed at all came from the cue ball.
    obj_speed = np.linalg.norm(result.velocities[1], axis=1)
    cue_speed = np.linalg.norm(result.velocities[0], axis=1)
    assert obj_speed.max() > 0.5 * cue_speed.max()
    assert np.abs(np.diff(result.trajectories[1][:, 0])).sum() > 0.5
