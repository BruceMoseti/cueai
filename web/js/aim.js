/**
 * Ghost-ball geometry.
 *
 * To send an object ball toward a pocket, the cue ball's centre at the moment
 * of contact has to sit on the line from the pocket through the object ball,
 * two radii back. That point is the "ghost ball", and aiming the cue centre at
 * it is exact rather than approximate: `tests/test_validation.py` asserts that
 * the shot drops and that half a degree either side of it misses.
 *
 * Everything here is closed form and costs microseconds, which is why the bot
 * spends its simulation budget on choosing between shots rather than on aiming
 * them.
 */

import { BALL } from "./physics.js";

const R = BALL.radius;

export function ghostBall(cue, obj, target) {
  const dx = target[0] - obj.x;
  const dy = target[1] - obj.y;
  const d = Math.hypot(dx, dy);
  if (d < 1e-9) return null;
  const gx = obj.x - (2 * R * dx) / d;
  const gy = obj.y - (2 * R * dy) / d;

  const ax = gx - cue.x;
  const ay = gy - cue.y;
  const aLen = Math.hypot(ax, ay);
  if (aLen < 1e-9) return null;

  // Cut angle between the cue ball's approach and the object ball's departure.
  const cut = Math.acos(Math.max(-1, Math.min(1, (ax / aLen) * (dx / d) + (ay / aLen) * (dy / d))));
  return {
    x: gx,
    y: gy,
    angle: Math.atan2(ay, ax),
    cut,
    cueTravel: aLen,
    objTravel: d,
  };
}

/** Perpendicular distance from a point to a segment, for blocker tests. */
export function pointSegmentDistance(px, py, ax, ay, bx, by) {
  const vx = bx - ax;
  const vy = by - ay;
  const len2 = vx * vx + vy * vy;
  if (len2 < 1e-12) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * vx + (py - ay) * vy) / len2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
}

/**
 * Is the corridor from (ax,ay) to (bx,by) wide enough for a ball to pass?
 *
 * A ball's centre has to stay two radii from any other centre, so the swept
 * corridor is blocked by anything closer than that to the centre line.
 */
export function pathClear(balls, ax, ay, bx, by, ignore = []) {
  for (const b of balls) {
    if (b.pocketed || ignore.includes(b.number)) continue;
    if (pointSegmentDistance(b.x, b.y, ax, ay, bx, by) < 2 * R - 1e-4) return false;
  }
  return true;
}

/**
 * Where the cue ball first makes contact, for the aiming overlay.
 *
 * Walks the aim line and returns the first ball whose centre comes within two
 * radii of it, plus the tangent and object-ball directions at that point.
 */
export function firstContact(balls, cue, angle, table) {
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);

  // Distance along the aim line until the cue ball reaches a rail.
  let limit = Infinity;
  if (dx > 1e-9) limit = Math.min(limit, (table.length - R - cue.x) / dx);
  if (dx < -1e-9) limit = Math.min(limit, (R - cue.x) / dx);
  if (dy > 1e-9) limit = Math.min(limit, (table.width - R - cue.y) / dy);
  if (dy < -1e-9) limit = Math.min(limit, (R - cue.y) / dy);
  if (!Number.isFinite(limit)) limit = 0;

  let best = null;
  for (const b of balls) {
    if (b.pocketed || b.number === 0) continue;
    // Solve |cue + t*d - b| = 2R for the smallest positive t.
    const ex = b.x - cue.x;
    const ey = b.y - cue.y;
    const proj = ex * dx + ey * dy;
    if (proj <= 0) continue;
    const perp2 = ex * ex + ey * ey - proj * proj;
    const gap = 4 * R * R - perp2;
    if (gap < 0) continue;
    const t = proj - Math.sqrt(gap);
    if (t < 0) continue;
    if (!best || t < best.t) best = { t, ball: b };
  }

  if (!best || best.t > limit) {
    return { hit: null, x: cue.x + limit * dx, y: cue.y + limit * dy, distance: limit };
  }

  const cx = cue.x + best.t * dx;
  const cy = cue.y + best.t * dy;
  // The object ball leaves along the line of centres; for a rolling cue ball
  // the tangent line is perpendicular to it. That 90 degree separation is the
  // rule players learn, and it falls out of the impulse being along n.
  const nx = (best.ball.x - cx) / (2 * R);
  const ny = (best.ball.y - cy) / (2 * R);
  return {
    hit: best.ball,
    x: cx,
    y: cy,
    distance: best.t,
    objectDir: [nx, ny],
    tangentDir: [-ny, nx],
  };
}
