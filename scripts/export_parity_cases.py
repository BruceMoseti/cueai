#!/usr/bin/env python3
"""
Export reference shots so the browser physics can be checked against Python.

The Python package under `src/pocket/physics/` is the definition of correct: it
is what `tests/test_validation.py` pins to closed-form mechanics. The browser
runs a hand port of it, and a port is only worth anything if someone measures
the difference. This writes the initial conditions and the reference outcome
for a spread of shots; `web/test/parity.mjs` replays them in Node.

Each case also carries a chaos yardstick. Break shots are Lyapunov unstable, so
two implementations that differ in the last bit of a float will not agree on
where sixteen balls come to rest, and demanding that they do would be a
misunderstanding rather than a standard. The yardstick is how far the reference
moves when its own initial condition is nudged by a picometre, which bounds how
much agreement is available to ask for.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pocket.physics.ball import Ball
from pocket.physics.constants import BallParams, ShotParams, TableParams
from pocket.physics.rack import identity_for, make_full_rack
from pocket.physics.simulator import Simulator

PERTURBATION_M = 1e-12


def make_ball(number: int, pos: tuple[float, float], params: BallParams) -> Ball:
    return Ball(
        id=number,
        number=number,
        pos=np.array(pos, dtype=np.float64),
        vel=np.zeros(2),
        omega=np.zeros(3),
        params=params,
        identity=identity_for(number),
    )


def scatter_layout(
    rng: np.random.Generator, table: TableParams, params: BallParams, n_objects: int
) -> list[Ball]:
    """Place a cue ball and a few object balls with no initial overlap."""
    R = params.radius
    margin = 3 * R
    placed: list[tuple[float, float]] = []
    while len(placed) < n_objects + 1:
        candidate = (
            float(rng.uniform(margin, table.length - margin)),
            float(rng.uniform(margin, table.width - margin)),
        )
        if any(np.hypot(candidate[0] - x, candidate[1] - y) < 4 * R for x, y in placed):
            continue
        # Keep them off the pocket mouths so the layout itself is not a pot.
        if any(np.hypot(candidate[0] - px, candidate[1] - py) < 3 * R for px, py in table.pockets):
            continue
        placed.append(candidate)

    numbers = [0, *rng.choice(np.arange(1, 16), size=n_objects, replace=False)]
    return [make_ball(int(n), pos, params) for n, pos in zip(numbers, placed)]


def serialize_balls(balls: list[Ball]) -> list[dict]:
    return [
        {"number": int(b.number), "x": float(b.pos[0]), "y": float(b.pos[1])} for b in balls
    ]


def outcome(sim: Simulator, balls: list[Ball], shot: ShotParams) -> dict:
    started = time.perf_counter()
    result = sim.simulate_shot(shot, balls=balls)
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "table_time": float(result.times[-1]) if len(result.times) else 0.0,
        "resting": {
            str(int(b.number)): [float(result.endpoints[b.id][0]), float(result.endpoints[b.id][1])]
            for b in balls
            if not result.pocketed[b.id]
        },
        "pocketed": sorted(int(b.number) for b in balls if result.pocketed[b.id]),
        "collisions": int(result.collision_events),
        "cushions": int(result.cushion_events),
    }


def chaos_yardstick(sim: Simulator, balls: list[Ball], shot: ShotParams, reference: dict) -> float:
    """
    How far the reference moves when its own input is nudged by a picometre.

    A porting difference smaller than this is indistinguishable from the
    sensitivity the system already has to the last bit of its inputs.
    """
    nudged = [b.copy() for b in balls]
    nudged[0].pos[0] += PERTURBATION_M
    other = outcome(sim, nudged, shot)
    worst = 0.0
    for number, pos in reference["resting"].items():
        if number not in other["resting"]:
            return float("inf")  # a ball changed pocketed status: fully chaotic
        elsewhere = other["resting"][number]
        moved = float(np.hypot(pos[0] - elsewhere[0], pos[1] - elsewhere[1]))
        worst = max(worst, moved)
    if reference["pocketed"] != other["pocketed"]:
        return float("inf")
    return worst


def build_cases(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    table = TableParams()
    params = BallParams()
    sim = Simulator(table=table, ball_params=params)
    cases: list[dict] = []

    # Hand-picked shots that exercise one mechanism each, so a parity failure
    # points at the part of the model that was ported wrong.
    named: list[tuple[str, list[Ball], ShotParams]] = [
        (
            "stun-into-rail",
            [make_ball(0, (0.6, 0.635), params)],
            ShotParams(speed=3.0, angle=0.0),
        ),
        (
            "draw",
            [make_ball(0, (0.6, 0.635), params), make_ball(1, (1.6, 0.635), params)],
            ShotParams(speed=3.5, angle=0.0, english_y=-0.45),
        ),
        (
            "follow",
            [make_ball(0, (0.6, 0.635), params), make_ball(1, (1.6, 0.635), params)],
            ShotParams(speed=3.5, angle=0.0, english_y=0.45),
        ),
        (
            "right-english-off-two-rails",
            [make_ball(0, (0.6, 0.4), params)],
            ShotParams(speed=4.5, angle=0.7, english_x=0.45),
        ),
        (
            "thin-cut",
            [make_ball(0, (0.6, 0.5), params), make_ball(1, (1.5, 0.7), params)],
            ShotParams(speed=4.0, angle=0.2),
        ),
        (
            "corner-pocket",
            [make_ball(0, (1.2, 0.635), params), make_ball(1, (2.0, 0.9), params)],
            ShotParams(speed=3.0, angle=float(np.arctan2(0.9 - 0.635, 2.0 - 1.2))),
        ),
        (
            "three-ball-cluster",
            [
                make_ball(0, (0.5, 0.635), params),
                make_ball(1, (1.5, 0.635), params),
                make_ball(2, (1.5 + 2 * params.radius + 1e-4, 0.66), params),
                make_ball(3, (1.5 + 2 * params.radius + 1e-4, 0.60), params),
            ],
            ShotParams(speed=5.0, angle=0.0),
        ),
        (
            "soft-roll",
            [make_ball(0, (0.5, 0.635), params)],
            ShotParams(speed=0.8, angle=0.35),
        ),
        (
            "masse-lite",
            [make_ball(0, (0.9, 0.5), params)],
            ShotParams(speed=2.5, angle=1.2, english_x=-0.4, cue_elevation=0.12),
        ),
    ]

    for name, balls, shot in named:
        ref = outcome(sim, balls, shot)
        cases.append(
            {
                "name": name,
                "balls": serialize_balls(balls),
                "shot": {
                    "speed": shot.speed,
                    "angle": shot.angle,
                    "english_x": shot.english_x,
                    "english_y": shot.english_y,
                    "cue_elevation": shot.cue_elevation,
                },
                "reference": ref,
                "chaos_yardstick_m": chaos_yardstick(sim, balls, shot, ref),
            }
        )

    # Randomised layouts, to catch anything the hand-picked shots miss.
    for i in range(24):
        balls = scatter_layout(rng, table, params, n_objects=int(rng.integers(1, 5)))
        shot = ShotParams(
            speed=float(rng.uniform(1.0, 6.0)),
            angle=float(rng.uniform(-np.pi, np.pi)),
            english_x=float(rng.uniform(-0.45, 0.45)),
            english_y=float(rng.uniform(-0.45, 0.45)),
        )
        ref = outcome(sim, balls, shot)
        cases.append(
            {
                "name": f"random-{i:02d}",
                "balls": serialize_balls(balls),
                "shot": {
                    "speed": shot.speed,
                    "angle": shot.angle,
                    "english_x": shot.english_x,
                    "english_y": shot.english_y,
                    "cue_elevation": 0.0,
                },
                "reference": ref,
                "chaos_yardstick_m": chaos_yardstick(sim, balls, shot, ref),
            }
        )

    # The break: sixteen balls, the case the game actually opens with.
    for speed in (6.0, 8.0):
        balls = make_full_rack(table=table, ball_params=params, seed=7)
        shot = ShotParams(speed=speed, angle=0.0)
        ref = outcome(sim, balls, shot)
        cases.append(
            {
                "name": f"break-{speed:.0f}ms",
                "balls": serialize_balls(balls),
                "shot": {
                    "speed": shot.speed,
                    "angle": shot.angle,
                    "english_x": 0.0,
                    "english_y": 0.0,
                    "cue_elevation": 0.0,
                },
                "reference": ref,
                "chaos_yardstick_m": chaos_yardstick(sim, balls, shot, ref),
            }
        )

    return cases


def physics_only(case: dict) -> dict:
    """The reproducible part of a case, with wall-clock timings dropped."""
    return {
        "name": case["name"],
        "balls": case["balls"],
        "shot": case["shot"],
        "pocketed": case["reference"]["pocketed"],
        "resting": case["reference"]["resting"],
    }


def check_against(path: Path, cases: list[dict]) -> int:
    """
    Verify the committed cases still describe what the simulator does.

    Timings vary run to run, so comparing the files byte for byte would fail
    for no reason; only the physics is compared.
    """
    if not path.exists():
        print(f"{path} does not exist; run without --check to create it")
        return 1

    committed = {c["name"]: physics_only(c) for c in json.loads(path.read_text())["cases"]}
    fresh = {c["name"]: physics_only(c) for c in cases}

    if committed.keys() != fresh.keys():
        print(f"case list changed: {sorted(set(fresh) ^ set(committed))}")
        return 1

    worst = 0.0
    worst_name = ""
    for name, current in fresh.items():
        old = committed[name]
        if old["pocketed"] != current["pocketed"]:
            print(f"{name}: pocketed {old['pocketed']} but now pockets {current['pocketed']}")
            return 1
        for number, pos in current["resting"].items():
            before = old["resting"][number]
            moved = float(np.hypot(pos[0] - before[0], pos[1] - before[1]))
            if moved > worst:
                worst, worst_name = moved, f"{name} ball {number}"

    if worst > 1e-6:
        print(f"committed cases are stale: {worst * 1000:.4f} mm drift on {worst_name}")
        print("run `make parity` and commit web/test/parity_cases.json")
        return 1
    print(f"committed cases match the simulator (worst drift {worst * 1000:.2e} mm)")
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("web/test/parity_cases.json"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed cases instead of rewriting them",
    )
    args = parser.parse_args(argv)

    cases = build_cases(args.seed)

    if args.check:
        raise SystemExit(check_against(args.out, cases))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dt": 0.001,
        "max_time": 15.0,
        "perturbation_m": PERTURBATION_M,
        "note": (
            "Generated by scripts/export_parity_cases.py from the Python reference "
            "simulator. chaos_yardstick_m is how far that reference moves under a "
            "1e-12 m nudge to the cue ball, which bounds the agreement any second "
            "implementation can be asked for."
        ),
        "cases": cases,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n")

    finite = [c["chaos_yardstick_m"] for c in cases if np.isfinite(c["chaos_yardstick_m"])]
    total = sum(c["reference"]["seconds"] for c in cases)
    table_time = sum(c["reference"]["table_time"] for c in cases)
    print(f"wrote {len(cases)} cases to {args.out}")
    print(f"  {len(finite)} deterministic enough to compare tightly")
    print(f"  {len(cases) - len(finite)} chaotic (a picometre changes which balls drop)")
    print(f"  reference spent {total:.2f} s simulating {table_time:.1f} s of table time")


if __name__ == "__main__":
    main()
