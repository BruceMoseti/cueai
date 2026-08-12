"""Full 8-ball rack layout and ball identities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cueai.physics.ball import Ball
from cueai.physics.constants import BallParams, TableParams

# Standard pool ball colors (RGB 0-255)
BALL_COLORS: dict[int, tuple[int, int, int]] = {
    0: (245, 245, 240),   # cue
    1: (235, 200, 40),    # yellow
    2: (40, 90, 200),     # blue
    3: (200, 40, 40),     # red
    4: (120, 50, 160),    # purple
    5: (230, 130, 30),    # orange
    6: (30, 140, 60),     # green
    7: (120, 40, 50),     # maroon
    8: (20, 20, 20),      # eight
    9: (235, 200, 40),
    10: (40, 90, 200),
    11: (200, 40, 40),
    12: (120, 50, 160),
    13: (230, 130, 30),
    14: (30, 140, 60),
    15: (120, 40, 50),
}


@dataclass(frozen=True)
class BallIdentity:
    number: int  # 0 = cue, 1-15 object
    suit: str    # cue | solid | stripe | eight

    @property
    def color(self) -> tuple[int, int, int]:
        return BALL_COLORS[self.number]

    @property
    def is_stripe(self) -> bool:
        return self.suit == "stripe"


def identity_for(number: int) -> BallIdentity:
    if number == 0:
        return BallIdentity(0, "cue")
    if number == 8:
        return BallIdentity(8, "eight")
    if 1 <= number <= 7:
        return BallIdentity(number, "solid")
    return BallIdentity(number, "stripe")


def foot_spot(table: TableParams) -> np.ndarray:
    """Apex of the rack (foot spot)."""
    return np.array([table.length * 0.75, table.width * 0.5], dtype=np.float64)


def head_spot(table: TableParams) -> np.ndarray:
    """Cue ball default (kitchen / head spot)."""
    return np.array([table.length * 0.25, table.width * 0.5], dtype=np.float64)


def triangle_positions(apex: np.ndarray, R: float) -> list[np.ndarray]:
    """
    15 tightly packed positions: row 0 has 1 ball (apex), … row 4 has 5.
    Oriented toward the head of the table (−x).
    """
    gap = 2.0 * R + 1e-4  # tiny clearance so they aren't already overlapping
    positions: list[np.ndarray] = []
    for row in range(5):
        for col in range(row + 1):
            # Back along −x, lateral along ±y
            x = apex[0] + row * gap * np.sqrt(3) / 2
            y = apex[1] + (col - row / 2.0) * gap
            positions.append(np.array([x, y], dtype=np.float64))
    return positions


def standard_rack_order(seed: int | None = 7) -> list[int]:
    """
    Legal-ish 8-ball rack:
      - apex = any (use 1)
      - 8-ball in the center of the rack (index 4 in triangle = row2 col1)
      - back corners: one solid, one stripe
    Remaining balls shuffled.
    """
    rng = np.random.default_rng(seed)
    solids = [1, 2, 3, 4, 5, 6, 7]
    stripes = [9, 10, 11, 12, 13, 14, 15]
    rng.shuffle(solids)
    rng.shuffle(stripes)

    order = [0] * 15
    # Triangle indices:
    # 0
    # 1 2
    # 3 4 5
    # 6 7 8 9
    # 10 11 12 13 14
    order[0] = solids.pop()          # apex
    order[4] = 8                     # center
    order[10] = solids.pop()         # back-left corner solid
    order[14] = stripes.pop()        # back-right corner stripe

    remaining = solids + stripes
    rng.shuffle(remaining)
    for i in range(15):
        if order[i] == 0:
            order[i] = remaining.pop()
    return order


def make_full_rack(
    table: TableParams | None = None,
    ball_params: BallParams | None = None,
    cue_pos: tuple[float, float] | None = None,
    seed: int | None = 7,
) -> list[Ball]:
    """Cue ball + 15 racked object balls."""
    table = table or TableParams()
    ball_params = ball_params or BallParams()
    R = ball_params.radius
    apex = foot_spot(table)
    spots = triangle_positions(apex, R)
    numbers = standard_rack_order(seed)

    balls: list[Ball] = []
    # Cue
    cp = np.array(cue_pos, dtype=np.float64) if cue_pos else head_spot(table)
    cue = Ball(
        id=0,
        number=0,
        pos=cp,
        vel=np.zeros(2),
        omega=np.zeros(3),
        params=ball_params,
        identity=identity_for(0),
    )
    balls.append(cue)

    for num, pos in zip(numbers, spots):
        balls.append(
            Ball(
                id=num,  # use ball number as id for clarity
                number=num,
                pos=pos.copy(),
                vel=np.zeros(2),
                omega=np.zeros(3),
                params=ball_params,
                identity=identity_for(num),
            )
        )
    return balls
