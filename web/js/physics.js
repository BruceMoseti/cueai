/**
 * Billiards physics, ported line for line from `src/cueai/physics/`.
 *
 * This is deliberately a transcription rather than a reimplementation: the
 * Python package is the reference that `tests/test_validation.py` checks
 * against closed-form mechanics, so the browser is only trustworthy insofar as
 * it reproduces it. `web/test/parity.mjs` runs both on the same shots and
 * fails if any resting position differs by more than a millimetre.
 *
 * Units are SI throughout: metres, seconds, radians.
 */

export const G = 9.81;

export const BALL = Object.freeze({
  radius: 0.028575, // m, a 2 1/4 inch ball
  mass: 0.17, // kg
  get inertia() {
    // Solid sphere, I = (2/5) m R^2.
    return 0.4 * this.mass * this.radius * this.radius;
  },
});

export function defaultTable() {
  const length = 2.54;
  const width = 1.27;
  return {
    length,
    width,
    muSlide: 0.2, // cloth sliding friction
    muRoll: 0.01, // rolling resistance
    muSpin: 0.044, // spinning (vertical axis) friction
    eBall: 0.95, // ball-ball restitution
    eCushion: 0.85,
    muBall: 0.06, // ball-ball tangential friction, the source of throw
    muCushion: 0.2,
    frictionNoiseAmp: 0.02, // cloth is not perfectly uniform
    frictionNoiseScale: 0.35, // correlation length, m
    pocketRadius: 0.06,
    pockets: [
      [0, 0],
      [length / 2, 0],
      [length, 0],
      [0, width],
      [length / 2, width],
      [length, width],
    ],
  };
}

export const MotionState = Object.freeze({
  SLIDING: "sliding",
  ROLLING: "rolling",
  SPINNING: "spinning",
  STATIONARY: "stationary",
  POCKETED: "pocketed",
});

export function makeBall(number, x, y) {
  return {
    number,
    x,
    y,
    vx: 0,
    vy: 0,
    // Angular velocity. wx/wy carry roll and draw, wz carries english.
    wx: 0,
    wy: 0,
    wz: 0,
    pocketed: false,
  };
}

/** Velocity of the cloth contact point: u = v + omega x (-R zhat). */
export function slipVelocity(b) {
  const R = BALL.radius;
  return [b.vx - b.wy * R, b.vy + b.wx * R];
}

export function speed(b) {
  return Math.hypot(b.vx, b.vy);
}

export function motionState(b, eps = 1e-3) {
  if (b.pocketed) return MotionState.POCKETED;
  const [ux, uy] = slipVelocity(b);
  const uMag = Math.hypot(ux, uy);
  const vMag = speed(b);
  const wz = Math.abs(b.wz);
  // The tolerance has to exceed one step's slip decrement (~3.5 mu g dt) or the
  // state flips back and forth across the transition.
  const slipEps = Math.max(0.01, 0.01 * vMag);
  if (vMag < eps && uMag < eps && wz < eps) return MotionState.STATIONARY;
  if (uMag < slipEps && vMag >= eps) return MotionState.ROLLING;
  if (vMag < eps && wz >= eps) return MotionState.SPINNING;
  return MotionState.SLIDING;
}

/**
 * Spin-down about the vertical axis, clamped at zero.
 *
 * The decrement per step is constant, so subtracting it unconditionally steps
 * past zero and flips the sign; the ball then chatters between two small
 * values and never reaches rest. Friction removes spin, it cannot reverse it.
 */
export function decaySpin(wz, table, dt) {
  const step = ((2.5 * table.muSpin * G) / BALL.radius) * dt;
  if (Math.abs(wz) <= step) return 0;
  return wz - Math.sign(wz) * step;
}

/** Smooth spatial variation in cloth friction, so the table is not ideal. */
export function localMuSlide(table, x, y) {
  const amp = table.frictionNoiseAmp;
  if (amp <= 0) return table.muSlide;
  const s = table.frictionNoiseScale;
  const nx = Math.sin((x / s) * 2 * Math.PI) * Math.cos((y / s) * 2 * Math.PI);
  const ny = Math.sin(((x + y) / s) * Math.PI);
  return Math.min(0.45, Math.max(0.05, table.muSlide + amp * 0.5 * (nx + ny)));
}

/**
 * One Euler step of cloth dynamics.
 *
 * While the contact point slips, friction acts opposite the slip velocity and
 * the matching torque spins the ball up, so slip decays 3.5x faster than the
 * centre of mass and the ball settles into rolling at 5/7 of its launch speed.
 */
export function integrateBall(b, table, dt) {
  if (b.pocketed) return;

  const R = BALL.radius;
  const m = BALL.mass;
  const I = BALL.inertia;
  const state = motionState(b);

  if (state === MotionState.STATIONARY) {
    b.vx = 0;
    b.vy = 0;
    b.wx = 0;
    b.wy = 0;
    b.wz = 0;
    return;
  }

  if (state === MotionState.SPINNING) {
    b.wz = decaySpin(b.wz, table, dt);
    return;
  }

  if (state === MotionState.SLIDING) {
    const [ux, uy] = slipVelocity(b);
    const uMag = Math.hypot(ux, uy);
    if (uMag < 1e-9) return;

    const muS = localMuSlide(table, b.x, b.y);
    const ax = -muS * G * (ux / uMag);
    const ay = -muS * G * (uy / uMag);
    // tau = (-R zhat) x F  =>  alpha = (R m a_y / I, -R m a_x / I, 0)
    const alphaX = (R * m * ay) / I;
    const alphaY = (-R * m * ax) / I;

    b.vx += ax * dt;
    b.vy += ay * dt;
    b.wx += alphaX * dt;
    b.wy += alphaY * dt;
    b.wz = decaySpin(b.wz, table, dt);

    const [u2x, u2y] = slipVelocity(b);
    if (Math.hypot(u2x, u2y) < Math.max(1e-3, 0.01 * speed(b))) {
      // Snap onto the rolling constraint rather than hovering just above it.
      b.wx = -b.vy / R;
      b.wy = b.vx / R;
    }
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    return;
  }

  const vMag = speed(b);
  if (vMag < 1e-9) {
    b.vx = 0;
    b.vy = 0;
    b.wx = 0;
    b.wy = 0;
    return;
  }
  // Rolling: only rolling resistance. Vertical-axis spin exerts no lateral
  // force on a rolling rigid sphere; english acts through cushions and throw.
  const ax = -table.muRoll * G * (b.vx / vMag);
  const ay = -table.muRoll * G * (b.vy / vMag);
  b.vx += ax * dt;
  b.vy += ay * dt;
  if (speed(b) < 2e-2) {
    b.vx = 0;
    b.vy = 0;
    b.wx = 0;
    b.wy = 0;
    b.wz = 0;
    return;
  }
  b.wx = -b.vy / R;
  b.wy = b.vx / R;
  b.wz = decaySpin(b.wz, table, dt);
  b.x += b.vx * dt;
  b.y += b.vy * dt;
}

/** Velocity-dependent ball-ball friction, Alciatore TP A-14 style. */
export function ballBallFriction(vRel, table) {
  const base = 0.02 + 0.08 * Math.exp(-0.85 * Math.abs(vRel));
  return Math.min(0.25, Math.max(0.01, 0.5 * (base + table.muBall)));
}

/** Equal-mass frictional collision with spin transfer and throw. */
export function resolveBallBall(a, b, table) {
  if (a.pocketed || b.pocketed) return false;

  const R = BALL.radius;
  const m = BALL.mass;
  const I = BALL.inertia;

  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dist = Math.hypot(dx, dy);
  const minDist = 2 * R;
  if (dist < 1e-12 || dist > minDist + 2e-4) return false;

  const nx = dx / dist;
  const ny = dy / dist;
  const overlap = minDist - dist;
  // Spin contributes nothing along n, so the approach speed can be tested
  // before any impulse work. Balls resting in contact stop here.
  const vN = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
  if (vN <= 1e-6 && overlap <= 0) return false;

  if (overlap > 0) {
    a.x -= 0.5 * overlap * nx;
    a.y -= 0.5 * overlap * ny;
    b.x += 0.5 * overlap * nx;
    b.y += 0.5 * overlap * ny;
  }
  if (vN <= 1e-6) return false; // separating or resting: position fixed only

  // Contact-point relative velocity, including both spins.
  const raX = R * nx;
  const raY = R * ny;
  const rbX = -R * nx;
  const rbY = -R * ny;
  // omega x r, with r in the plane and omega fully three-dimensional.
  const crossA = [a.wy * 0 - a.wz * raY, a.wz * raX - a.wx * 0];
  const crossB = [b.wy * 0 - b.wz * rbY, b.wz * rbX - b.wx * 0];
  const relX = a.vx - b.vx + crossA[0] - crossB[0];
  const relY = a.vy - b.vy + crossA[1] - crossB[1];

  const speedN = Math.abs(vN);
  const eEff = Math.min(0.98, Math.max(0.75, table.eBall - 0.02 * Math.max(0, speedN - 2)));
  const jN = -(1 + eEff) * vN * (m / 2);

  let tX = relX - vN * nx;
  let tY = relY - vN * ny;
  const vT = Math.hypot(tX, tY);
  if (vT > 1e-9) {
    tX /= vT;
    tY /= vT;
  } else {
    tX = -ny;
    tY = nx;
  }

  const muB = ballBallFriction(vT + speedN, table);
  const jTMax = muB * Math.abs(jN);
  const invMassT = 2 / m + (2 * R * R) / I;
  let jT = -vT / invMassT;
  jT = Math.min(jTMax, Math.max(-jTMax, jT));

  const jX = jN * nx + jT * tX;
  const jY = jN * ny + jT * tY;
  a.vx += jX / m;
  a.vy += jY / m;
  b.vx -= jX / m;
  b.vy -= jY / m;

  // r x J for an in-plane r and J only has a z component.
  a.wz += (raX * jY - raY * jX) / I;
  b.wz += (rbX * -jY - rbY * -jX) / I;
  // Residual vertical-axis coupling: cling and throw.
  a.wz += (0.15 * jT) / (I / R);
  b.wz -= (0.15 * jT) / (I / R);
  return true;
}

/** Cushion bounce with friction, spin transfer, and corner dual-rail hits. */
export function resolveCushion(b, table) {
  if (b.pocketed) return false;

  const R = BALL.radius;
  const m = BALL.mass;
  const I = BALL.inertia;
  const normals = [];

  if (b.x - R < 0) {
    b.x = R;
    normals.push([1, 0]);
  } else if (b.x + R > table.length) {
    b.x = table.length - R;
    normals.push([-1, 0]);
  }
  if (b.y - R < 0) {
    b.y = R;
    normals.push([0, 1]);
  } else if (b.y + R > table.width) {
    b.y = table.width - R;
    normals.push([0, -1]);
  }
  if (normals.length === 0) return false;

  let hit = false;
  for (const [nx, ny] of normals) {
    // Near a pocket mouth the ball should drop, not rebound off the jaw.
    let nearPocket = false;
    for (const [px, py] of table.pockets) {
      if (Math.hypot(b.x - px, b.y - py) < table.pocketRadius * 1.15) {
        nearPocket = true;
        break;
      }
    }
    if (nearPocket) continue;

    const vN = b.vx * nx + b.vy * ny;
    if (vN >= 0) continue;

    const tX = -ny;
    const tY = nx;
    // Contact point sits against the rail: r = -R n.
    const rX = -R * nx;
    const rY = -R * ny;
    const contactX = b.vx + (b.wy * 0 - b.wz * rY);
    const contactY = b.vy + (b.wz * rX - b.wx * 0);
    const vTContact = contactX * tX + contactY * tY;

    const eEff = Math.min(0.92, Math.max(0.55, table.eCushion - 0.03 * Math.max(0, Math.abs(vN) - 1.5)));
    const jN = -(1 + eEff) * vN * m;

    const invMassT = 1 / m + (R * R) / I;
    let jT = -vTContact / invMassT;
    const jTMax = table.muCushion * Math.abs(jN);
    jT = Math.min(jTMax, Math.max(-jTMax, jT));

    const jX = jN * nx + jT * tX;
    const jY = jN * ny + jT * tY;
    b.vx += jX / m;
    b.vy += jY / m;
    b.wz += (rX * jY - rY * jX) / I;
    b.wz *= 0.85; // the rail scrubs off some english
    hit = true;
  }
  return hit;
}

export function checkPocket(b, table) {
  if (b.pocketed) return false;
  for (const [px, py] of table.pockets) {
    if (Math.hypot(b.x - px, b.y - py) < table.pocketRadius) {
      b.pocketed = true;
      b.vx = 0;
      b.vy = 0;
      b.wx = 0;
      b.wy = 0;
      b.wz = 0;
      b.x = -1;
      b.y = -1;
      return true;
    }
  }
  return false;
}

/**
 * Multi-pass pairwise resolution.
 *
 * A single pass cannot propagate an impulse through a packed rack, where one
 * contact pushes a ball into the next. Deepest overlaps are resolved first.
 */
export function resolveAllBallCollisions(balls, table, events = null, passes = 20) {
  for (let pass = 0; pass < passes; pass++) {
    const active = [];
    for (let i = 0; i < balls.length; i++) if (!balls[i].pocketed) active.push(i);
    if (active.length < 2) break;

    const pairs = [];
    for (let ai = 0; ai < active.length; ai++) {
      for (let bi = ai + 1; bi < active.length; bi++) {
        const i = active[ai];
        const j = active[bi];
        const gap = Math.hypot(balls[j].x - balls[i].x, balls[j].y - balls[i].y) - 2 * BALL.radius;
        if (gap <= 1e-4) pairs.push([gap, i, j]);
      }
    }
    // Matches the reference: only an empty candidate set ends the sweep. A pass
    // that merely separates resting balls still has to be followed by another.
    if (pairs.length === 0) break;
    pairs.sort((p, q) => p[0] - q[0]);
    for (const [, i, j] of pairs) {
      if (!resolveBallBall(balls[i], balls[j], table)) continue;
      if (!events) continue;
      events.collisions++;
      // The first object ball the cue touches decides whether the shot is legal.
      if (events.firstContact === null) {
        if (balls[i].number === 0) events.firstContact = balls[j].number;
        else if (balls[j].number === 0) events.firstContact = balls[i].number;
      }
    }
  }
}

/**
 * Advance the whole table by one timestep.
 *
 * The order matches `Simulator.run`: integrate and handle rails per ball, then
 * resolve the cluster, then re-check rails and pockets because the cluster
 * shove can push a ball into either.
 */
export function stepWorld(balls, table, dt, events = null) {
  for (const b of balls) {
    if (b.pocketed) continue;
    integrateBall(b, table, dt);
    if (resolveCushion(b, table) && events) {
      events.cushions++;
      if (events.firstContact !== null) events.railAfterContact = true;
    }
    if (checkPocket(b, table) && events) events.potted.push(b.number);
  }

  resolveAllBallCollisions(balls, table, events);

  for (const b of balls) {
    if (b.pocketed) continue;
    if (resolveCushion(b, table) && events) {
      events.cushions++;
      if (events.firstContact !== null) events.railAfterContact = true;
    }
    if (checkPocket(b, table) && events) events.potted.push(b.number);
  }
}

export function newEvents() {
  return {
    potted: [],
    collisions: 0,
    cushions: 0,
    firstContact: null,
    railAfterContact: false,
  };
}

/**
 * Run a shot to completion without rendering, for the bot's search.
 *
 * Mirrors the reference loop, including the requirement that the table stay at
 * rest for a while before the shot is called over.
 */
export function simulateToRest(balls, table, { dt = 0.002, maxTime = 10 } = {}) {
  const events = newEvents();
  let restSteps = 0;
  const steps = Math.floor(maxTime / dt);
  let step = 0;
  for (; step < steps; step++) {
    stepWorld(balls, table, dt, events);
    if (anyBallMoving(balls)) {
      restSteps = 0;
    } else {
      restSteps++;
      if (restSteps > 25 && step > 20) break;
    }
  }
  events.tableTime = step * dt;
  return events;
}

export function anyBallMoving(balls) {
  for (const b of balls) {
    if (b.pocketed) continue;
    if (speed(b) > 1e-4 || Math.hypot(b.wx, b.wy, b.wz) > 1e-3) return true;
  }
  return false;
}

/**
 * Cue strike: tip offsets map to spin as omega = 2.5 * f * v / R.
 *
 * A horizontal impulse J applied a distance f*R off centre gives dv = J/m and
 * dw = J f R / I, and with I = (2/5) m R^2 that ratio is 2.5 f v / R. So
 * f = 0.4 launches the ball already rolling, and |f| > 0.5 miscues.
 */
export function applyShot(cue, { speed: v, angle, englishX = 0, englishY = 0 }) {
  const R = BALL.radius;
  cue.vx = v * Math.cos(angle);
  cue.vy = v * Math.sin(angle);
  // Top/backspin acts about the horizontal axis perpendicular to travel.
  cue.wx = 2.5 * englishY * (v / R) * -Math.sin(angle);
  cue.wy = 2.5 * englishY * (v / R) * Math.cos(angle);
  // Right english is clockwise seen from above, hence negative wz.
  cue.wz = -2.5 * englishX * (v / R);
}

export const MAX_TIP_OFFSET = 0.5; // beyond this a real stroke miscues
