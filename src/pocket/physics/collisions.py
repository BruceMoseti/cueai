"""Ball-ball and ball-cushion collisions with friction, throw, and multi-hit."""

from __future__ import annotations

import numpy as np

from pocket.physics.ball import Ball
from pocket.physics.constants import TableParams

# Two balls count as touching once their surfaces are inside this band.
#
# The value is a constant rather than a literal because it has a correctness
# requirement, not just a tuning one: it must sit far above the floating-point
# noise in a position (~1e-16 m) and far below anything physical, and it must
# not coincide with the gap balls are racked at. It used to be 1e-4, exactly
# the racking clearance, and the consequence was that only sixteen of a rack's
# thirty contacts registered — which sixteen being decided by whether hypot()
# happened to round up or down. A break then propagated through a contact graph
# with holes in it, so balls in the middle of the rack came out of a full-power
# break having never moved.
CONTACT_BAND = 1e-5


def ball_ball_friction(v_rel: float, table: TableParams) -> float:
    """
    Velocity-dependent ball-ball friction (Alciatore TP A-14 style).
    μ_b ≈ a + b exp(-c |v_rel|)
    """
    a, b, c = 0.02, 0.08, 0.85
    base = a + b * np.exp(-c * abs(v_rel))
    # Scale toward configured table.mu_ball
    return float(np.clip(0.5 * (base + table.mu_ball), 0.01, 0.25))


def resolve_ball_ball(a: Ball, b: Ball, table: TableParams) -> tuple[Ball, Ball]:
    """
    Equal-mass frictional inelastic collision with spin transfer / throw.

    Includes relative-velocity-dependent friction and contact-point ω coupling.
    """
    if a.pocketed or b.pocketed:
        return a, b

    delta = b.pos - a.pos
    dist = float(np.hypot(delta[0], delta[1]))
    min_dist = a.R + b.R
    if dist < 1e-12 or dist > min_dist + CONTACT_BAND:
        return a, b

    n = delta / dist
    overlap = min_dist - dist
    # Angular velocity contributes nothing along n, because ω × (R n) ⊥ n, so the
    # approach speed can be tested before doing any of the impulse work. Balls
    # resting in contact — most pairs in a packed rack, every step — stop here.
    v_n = float((a.vel - b.vel) @ n)
    if v_n <= 1e-6 and overlap <= 0:
        return a, b

    a, b = a.copy(), b.copy()
    if overlap > 0:
        # Mass-proportional separation (equal mass → half/half)
        a.pos = a.pos - 0.5 * overlap * n
        b.pos = b.pos + 0.5 * overlap * n
    if v_n <= 1e-6:
        return a, b  # separating or resting: positional correction only

    ra = a.R * np.array([n[0], n[1], 0.0])
    rb = -b.R * np.array([n[0], n[1], 0.0])
    wa = np.array([a.omega[0], a.omega[1], a.omega[2]])
    wb = np.array([b.omega[0], b.omega[1], b.omega[2]])
    va3 = np.array([a.vel[0], a.vel[1], 0.0])
    vb3 = np.array([b.vel[0], b.vel[1], 0.0])
    v_rel = (va3 - vb3) + np.cross(wa, ra) - np.cross(wb, rb)

    # v_n was already established above from the linear velocities; the spin terms
    # in v_rel are tangential and only matter for the friction impulse below.
    e = table.e_ball
    # Slightly softer at high speed (energy loss grows)
    speed_n = abs(v_n)
    e_eff = float(np.clip(e - 0.02 * max(0.0, speed_n - 2.0), 0.75, 0.98))
    m = a.m
    # J < 0 pushes a opposite n and b along n
    j_n = -(1 + e_eff) * v_n * (m / 2.0)

    v_t_vec = v_rel[:2] - v_n * n
    # Include out-of-plane spin contribution projected onto tangent
    v_t = float(np.linalg.norm(v_t_vec))
    if v_t > 1e-9:
        t_hat = v_t_vec / v_t
    else:
        t_hat = np.array([-n[1], n[0]])

    mu_b = ball_ball_friction(v_t + speed_n, table)
    j_t_max = mu_b * abs(j_n)
    I = a.I
    R = a.R
    inv_m_t = 2.0 / m + 2.0 * (R**2) / I
    j_t = -v_t / inv_m_t
    j_t = float(np.clip(j_t, -j_t_max, j_t_max))

    J = j_n * n + j_t * t_hat
    a.vel = a.vel + J / m
    b.vel = b.vel - J / m

    J3 = np.array([J[0], J[1], 0.0])
    a.omega = a.omega + np.cross(ra, J3) / I
    b.omega = b.omega + np.cross(rb, -J3) / I

    # Vertical-axis spin coupling (throw / cling residual)
    a.omega[2] += 0.15 * j_t / (I / R)
    b.omega[2] -= 0.15 * j_t / (I / R)
    return a, b


def resolve_cushion(ball: Ball, table: TableParams) -> Ball:
    """
    Cushion bounce with friction, spin transfer, and corner dual-rail hits.
    """
    b = ball.copy()
    if b.pocketed:
        return b

    R = b.R
    L, W = table.length, table.width
    e = table.e_cushion
    mu = table.mu_cushion

    # Detect which rails are penetrated (corners → both)
    normals: list[np.ndarray] = []
    if b.pos[0] - R < 0:
        b.pos[0] = R
        normals.append(np.array([1.0, 0.0]))
    elif b.pos[0] + R > L:
        b.pos[0] = L - R
        normals.append(np.array([-1.0, 0.0]))
    if b.pos[1] - R < 0:
        b.pos[1] = R
        normals.append(np.array([0.0, 1.0]))
    elif b.pos[1] + R > W:
        b.pos[1] = W - R
        normals.append(np.array([0.0, -1.0]))

    if not normals:
        return b

    for n in normals:
        # Skip if near a pocket mouth (ball should fall, not bounce hard)
        near_pocket = False
        for px, py in table.pockets:
            if float(np.hypot(b.pos[0] - px, b.pos[1] - py)) < table.pocket_radius * 1.15:
                near_pocket = True
                break
        if near_pocket:
            continue

        v_n = float(np.dot(b.vel, n))
        if v_n >= 0:
            continue

        t = np.array([-n[1], n[0]])
        # Contact velocity includes topspin/backspin into rail and sidespin along rail
        # ω × r_contact with r ≈ -R n (contact toward cushion)
        r_c = -R * np.array([n[0], n[1], 0.0])
        w3 = np.array([b.omega[0], b.omega[1], b.omega[2]])
        v_contact = np.array([b.vel[0], b.vel[1], 0.0]) + np.cross(w3, r_c)
        v_t_contact = float(np.dot(v_contact[:2], t))

        # Speed-dependent restitution (softer at high speed)
        e_eff = float(np.clip(e - 0.03 * max(0.0, abs(v_n) - 1.5), 0.55, 0.92))
        j_n = -(1 + e_eff) * v_n * b.m

        # Tangential friction + compliance-ish (partial stick)
        inv_m_t = 1.0 / b.m + (R**2) / b.I
        j_t = -v_t_contact / inv_m_t
        j_t = float(np.clip(j_t, -mu * abs(j_n), mu * abs(j_n)))

        b.vel = b.vel + (j_n * n + j_t * t) / b.m
        # Torques from rail impulse
        J3 = np.array([j_n * n[0] + j_t * t[0], j_n * n[1] + j_t * t[1], 0.0])
        b.omega = b.omega + np.cross(r_c, J3) / b.I
        # Rail compresses english somewhat
        b.omega[2] *= 0.85

    return b


def check_pocket(ball: Ball, table: TableParams) -> Ball:
    b = ball.copy()
    for px, py in table.pockets:
        # Slightly larger capture when moving toward pocket
        capture = table.pocket_radius
        if float(np.hypot(b.pos[0] - px, b.pos[1] - py)) < capture:
            b.pocketed = True
            b.vel[:] = 0
            b.omega[:] = 0
            # Park off-table visually
            b.pos = np.array([-1.0, -1.0], dtype=np.float64)
            break
    return b


def resolve_all_ball_collisions(
    balls: list[Ball], table: TableParams, passes: int = 24
) -> list[Ball]:
    """
    Multi-pass pairwise resolution so cluster breaks / simultaneous contacts
    propagate (critical for a packed rack).

    The count is a safety cap, not the usual cost: the loop exits as soon as a
    pass changes nothing, which for a table that is merely resting is the first
    one. It has to exceed the depth of the contact chain the impulse travels
    along, five rows for a full rack, and `web/js/physics.js` must use the same
    number or the two implementations diverge on exactly the shots that matter.
    """
    balls = [b.copy() for b in balls]
    n = len(balls)
    for _ in range(passes):
        any_hit = False
        # Candidate search is vectorised: with 16 balls this runs every time step,
        # so the pairwise distances are one numpy call rather than 120 Python ones.
        active = [i for i in range(n) if not balls[i].pocketed]
        if len(active) < 2:
            break
        positions = np.array([balls[i].pos for i in active])
        radii = np.array([balls[i].R for i in active])
        gaps = (
            np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
            - radii[:, None]
            - radii[None, :]
        )
        rows, cols = np.nonzero(np.triu(gaps <= CONTACT_BAND, k=1))
        # Process deepest overlaps first for stability
        pairs = sorted(
            (float(gaps[r, c]), active[r], active[c]) for r, c in zip(rows, cols)
        )
        for _, i, j in pairs:
            bi, bj = resolve_ball_ball(balls[i], balls[j], table)
            # Exact comparison, not ``allclose``: its 1e-8 absolute tolerance
            # would end the sweep a pass early on nanometre-scale corrections,
            # and since the JavaScript port reports what it did rather than
            # inferring it, a tolerance here is a divergence between the two.
            if (
                not np.array_equal(bi.vel, balls[i].vel)
                or not np.array_equal(bj.vel, balls[j].vel)
                or not np.array_equal(bi.pos, balls[i].pos)
                or not np.array_equal(bj.pos, balls[j].pos)
            ):
                any_hit = True
            balls[i], balls[j] = bi, bj
        # A pass that changed nothing leaves the next one the same problem, so
        # this is convergence rather than a budget. It matters now that a
        # racked table reports thirty resting contacts every step instead of
        # sixteen: without it every step below would pay for all the passes.
        if not any_hit:
            break
    return balls
