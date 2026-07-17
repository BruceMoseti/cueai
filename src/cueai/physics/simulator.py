"""High-level billiards simulator: full rack, multi-collision, spin cloth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cueai.physics.ball import Ball, integrate_ball
from cueai.physics.collisions import (
    check_pocket,
    resolve_all_ball_collisions,
    resolve_cushion,
)
from cueai.physics.constants import BallParams, ShotParams, TableParams
from cueai.physics.rack import make_full_rack


@dataclass
class SimResult:
    times: np.ndarray
    trajectories: dict[int, np.ndarray]  # id -> (T, 2) positions
    velocities: dict[int, np.ndarray]
    omegas: dict[int, np.ndarray]
    pocketed: dict[int, bool]
    features: np.ndarray
    endpoints: dict[int, np.ndarray]
    ball_meta: dict[int, dict] = field(default_factory=dict)
    collision_events: int = 0


@dataclass
class Simulator:
    table: TableParams = field(default_factory=TableParams)
    ball_params: BallParams = field(default_factory=BallParams)
    dt: float = 0.001
    max_time: float = 15.0
    collision_passes: int = 20

    def rack_full(
        self,
        cue_pos: tuple[float, float] | None = None,
        seed: int | None = 7,
    ) -> list[Ball]:
        return make_full_rack(
            table=self.table,
            ball_params=self.ball_params,
            cue_pos=cue_pos,
            seed=seed,
        )

    def rack_cue_and_object(
        self,
        cue_pos: tuple[float, float] | None = None,
        obj_pos: tuple[float, float] | None = None,
    ) -> list[Ball]:
        """Legacy 2-ball setup (kept for ML dataset compatibility)."""
        from cueai.physics.rack import identity_for

        L, W = self.table.length, self.table.width
        if cue_pos is None:
            cue_pos = (L * 0.25, W * 0.5)
        if obj_pos is None:
            obj_pos = (L * 0.65, W * 0.5)
        cue = Ball(
            id=0,
            number=0,
            pos=np.array(cue_pos, dtype=np.float64),
            vel=np.zeros(2),
            omega=np.zeros(3),
            params=self.ball_params,
            identity=identity_for(0),
        )
        obj = Ball(
            id=1,
            number=1,
            pos=np.array(obj_pos, dtype=np.float64),
            vel=np.zeros(2),
            omega=np.zeros(3),
            params=self.ball_params,
            identity=identity_for(1),
        )
        return [cue, obj]

    def apply_shot(self, cue: Ball, shot: ShotParams) -> Ball:
        c = cue.copy()
        c.vel = np.array(
            [shot.speed * np.cos(shot.angle), shot.speed * np.sin(shot.angle)],
            dtype=np.float64,
        )
        c.omega = shot.initial_omega(self.ball_params)
        # Elevation adds a bit of vertical-axis / massé-lite spin
        if abs(shot.cue_elevation) > 1e-6:
            c.omega[2] += 0.35 * shot.cue_elevation * shot.speed / self.ball_params.radius
        return c

    def run(self, balls: list[Ball], record_every: int = 4) -> SimResult:
        balls = [b.copy() for b in balls]
        t = 0.0
        step = 0
        times: list[float] = []
        traj: dict[int, list[np.ndarray]] = {b.id: [] for b in balls}
        vels: dict[int, list[np.ndarray]] = {b.id: [] for b in balls}
        omgs: dict[int, list[np.ndarray]] = {b.id: [] for b in balls}
        collisions = 0
        rest_steps = 0

        meta = {
            b.id: {
                "number": b.number,
                "suit": b.identity.suit if b.identity else "unknown",
                "color": list(b.identity.color) if b.identity else [200, 200, 200],
                "stripe": bool(b.identity and b.identity.is_stripe),
            }
            for b in balls
        }

        while t < self.max_time:
            moving = False
            for i, b in enumerate(balls):
                if b.pocketed:
                    continue
                balls[i] = integrate_ball(b, self.table, self.dt)
                balls[i] = resolve_cushion(balls[i], self.table)
                balls[i] = check_pocket(balls[i], self.table)
                if balls[i].speed() > 1e-4 or float(np.linalg.norm(balls[i].omega)) > 1e-3:
                    moving = True

            before = [(bb.vel.copy(), bb.omega.copy()) for bb in balls]
            balls = resolve_all_ball_collisions(
                balls, self.table, passes=self.collision_passes
            )
            for i, bb in enumerate(balls):
                if not np.allclose(before[i][0], bb.vel, atol=1e-6):
                    collisions += 1

            # Second cushion pass after cluster shove
            for i, b in enumerate(balls):
                if b.pocketed:
                    continue
                balls[i] = resolve_cushion(b, self.table)
                balls[i] = check_pocket(balls[i], self.table)

            if step % record_every == 0:
                times.append(t)
                for b in balls:
                    traj[b.id].append(b.pos.copy())
                    vels[b.id].append(b.vel.copy())
                    omgs[b.id].append(b.omega.copy())

            if not moving:
                rest_steps += 1
                if rest_steps > 25 and step > 20:
                    break
            else:
                rest_steps = 0

            t += self.dt
            step += 1

        trajectories = {k: np.asarray(v) for k, v in traj.items()}
        velocities = {k: np.asarray(v) for k, v in vels.items()}
        omegas = {k: np.asarray(v) for k, v in omgs.items()}
        endpoints = {
            k: (v[-1] if len(v) else np.zeros(2))
            for k, v in trajectories.items()
        }
        # Restore last on-table position for pocketed balls in endpoints
        for b in balls:
            if b.pocketed and b.id in trajectories and len(trajectories[b.id]):
                path = trajectories[b.id]
                on_table = path[np.all(path >= 0, axis=1)]
                if len(on_table):
                    endpoints[b.id] = on_table[-1]

        pocketed = {b.id: b.pocketed for b in balls}
        features = np.array(
            [
                endpoints.get(0, np.zeros(2))[0],
                endpoints.get(0, np.zeros(2))[1],
                float(sum(1 for v in pocketed.values() if v)),
                float(collisions),
                float(len(times)),
                self.table.mu_slide,
                self.table.friction_noise_amp,
            ],
            dtype=np.float64,
        )

        return SimResult(
            times=np.asarray(times),
            trajectories=trajectories,
            velocities=velocities,
            omegas=omegas,
            pocketed=pocketed,
            features=features,
            endpoints=endpoints,
            ball_meta=meta,
            collision_events=collisions,
        )

    def simulate_shot(
        self,
        shot: ShotParams,
        cue_pos: tuple[float, float] | None = None,
        obj_pos: tuple[float, float] | None = None,
        object_ball: bool = True,
        full_rack: bool = False,
        seed: int | None = 7,
        balls: list[Ball] | None = None,
    ) -> SimResult:
        if balls is not None:
            state = [b.copy() for b in balls]
        elif full_rack:
            state = self.rack_full(cue_pos=cue_pos, seed=seed)
        else:
            state = self.rack_cue_and_object(cue_pos, obj_pos)
            if not object_ball:
                state = [state[0]]
        # Find cue (id 0)
        for i, b in enumerate(state):
            if b.id == 0 or b.number == 0:
                state[i] = self.apply_shot(b, shot)
                break
        return self.run(state)


def shot_feature_vector(
    shot: ShotParams,
    cue_pos: np.ndarray,
    obj_pos: np.ndarray | None,
    table: TableParams,
) -> np.ndarray:
    """Input features for the ML residual predictor (pre-simulation)."""
    ox, oy = (obj_pos[0], obj_pos[1]) if obj_pos is not None else (0.0, 0.0)
    return np.array(
        [
            shot.speed,
            shot.angle,
            shot.english_x,
            shot.english_y,
            shot.cue_elevation,
            cue_pos[0],
            cue_pos[1],
            ox,
            oy,
            table.mu_slide,
            table.mu_roll,
            table.mu_spin,
            table.e_cushion,
            table.friction_noise_amp,
            np.cos(shot.angle) * shot.speed,
            np.sin(shot.angle) * shot.speed,
            shot.english_x * shot.speed,
            shot.english_y * shot.speed,
        ],
        dtype=np.float64,
    )


FEATURE_NAMES = [
    "speed",
    "angle",
    "english_x",
    "english_y",
    "cue_elevation",
    "cue_x",
    "cue_y",
    "obj_x",
    "obj_y",
    "mu_slide",
    "mu_roll",
    "mu_spin",
    "e_cushion",
    "friction_noise_amp",
    "vx0",
    "vy0",
    "spin_side_x_v",
    "spin_top_x_v",
]
