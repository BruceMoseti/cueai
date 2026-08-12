/**
 * Eight-ball rules.
 *
 * Pure state and decisions, no rendering and no timers, so the same code that
 * runs the visible game also runs inside the bot's search when it asks what a
 * candidate shot would leave behind.
 */

import { BALL, defaultTable, makeBall } from "./physics.js";
import { headSpot, makeRack, suitOf } from "./rack.js";

export const YOU = "you";
export const BOT = "bot";

export function opponentOf(player) {
  return player === YOU ? BOT : YOU;
}

export function createGame(seed = Math.floor(Math.random() * 1e9)) {
  const table = defaultTable();
  return {
    table,
    seed,
    balls: makeRack(table, seed),
    turn: YOU,
    groups: { [YOU]: null, [BOT]: null },
    phase: "break",
    ballInHand: true,
    behindHeadString: true, // the break must be struck from the kitchen
    winner: null,
    loseReason: null,
    shotCount: 0,
  };
}

export function ballsOf(state, group) {
  return state.balls.filter((b) => !b.pocketed && b.number !== 0 && suitOf(b.number) === group);
}

export function groupCleared(state, player) {
  const group = state.groups[player];
  if (!group) return false;
  return ballsOf(state, group).length === 0;
}

/** Which balls this player is allowed to hit first. */
export function legalTargets(state, player) {
  const onTable = state.balls.filter((b) => !b.pocketed && b.number !== 0);
  if (state.phase === "break") return onTable;
  const group = state.groups[player];
  if (!group) return onTable.filter((b) => b.number !== 8); // open table
  if (groupCleared(state, player)) return onTable.filter((b) => b.number === 8);
  return onTable.filter((b) => suitOf(b.number) === group);
}

/**
 * Apply the outcome of a completed shot to the game state.
 *
 * `events` comes straight from the physics; `wasBreak` and `clearedBefore` are
 * read from the state as it stood before the balls moved, because whether the
 * eight was legal depends on what was still on the table when it was struck.
 */
export function resolveShot(state, events, context) {
  const shooter = state.turn;
  const other = opponentOf(shooter);
  const potted = events.potted;
  const objectPotted = potted.filter((n) => n !== 0);
  const scratched = potted.includes(0);
  const eightPotted = potted.includes(8);
  const group = state.groups[shooter];

  const fouls = [];

  if (events.firstContact === null) {
    fouls.push("the cue ball hit nothing");
  } else if (context.wasBreak) {
    // Any first contact is legal on the break.
  } else if (group === null) {
    if (events.firstContact === 8) fouls.push("hit the 8 first with the table open");
  } else if (context.clearedBefore) {
    if (events.firstContact !== 8) fouls.push("had to hit the 8 first");
  } else if (suitOf(events.firstContact) !== group) {
    const wrong = suitOf(events.firstContact);
    fouls.push(wrong === "eight" ? "hit the 8 first" : `hit a ${wrong} first`);
  }

  if (context.wasBreak) {
    // A legal break drives four balls to a rail or pots one.
    if (potted.length === 0 && events.cushions < 4) fouls.push("the break did not reach four rails");
  } else if (potted.length === 0 && !events.railAfterContact) {
    fouls.push("no ball reached a rail after contact");
  }

  if (scratched) fouls.push("scratch");

  const foul = fouls.length > 0;

  // Assign groups on the first legal pot after the break.
  let assigned = null;
  if (!foul && !context.wasBreak && group === null && objectPotted.length > 0) {
    const first = objectPotted.find((n) => n !== 8);
    if (first !== undefined) {
      assigned = suitOf(first);
      state.groups[shooter] = assigned;
      state.groups[other] = assigned === "solid" ? "stripe" : "solid";
    }
  }

  if (eightPotted) {
    state.phase = "over";
    const legal = !foul && context.clearedBefore;
    state.winner = legal ? shooter : other;
    state.loseReason = legal
      ? null
      : foul
        ? `the 8 went down on a foul: ${fouls[0]}`
        : "the 8 went down before the group was cleared";
    return { foul, fouls, potted, objectPotted, assigned, continues: false, gameOver: true };
  }

  // A scratched cue ball stays off the table until the incoming player places
  // it, so there is never a moment where it exists at no legal position.
  state.phase = "play";
  state.shotCount++;

  // The shooter keeps the table only by legally potting one of their own.
  const ownPotted = objectPotted.filter((n) => {
    const g = state.groups[shooter];
    return g ? suitOf(n) === g : true;
  });
  const continues = !foul && ownPotted.length > 0;

  state.behindHeadString = false; // the kitchen restriction only binds the break
  if (!continues) {
    state.turn = other;
    state.ballInHand = foul;
  } else {
    state.ballInHand = false;
  }

  return { foul, fouls, potted, objectPotted, assigned, continues, gameOver: false };
}

/** Is this a legal place to drop the cue ball during ball in hand? */
export function canPlaceCue(state, x, y) {
  const R = BALL.radius;
  const t = state.table;
  if (x < R || x > t.length - R || y < R || y > t.width - R) return false;
  if (state.behindHeadString && x > t.length * 0.25) return false;
  for (const b of state.balls) {
    if (b.pocketed || b.number === 0) continue;
    if (Math.hypot(b.x - x, b.y - y) < 2 * R + 1e-3) return false;
  }
  for (const [px, py] of t.pockets) {
    if (Math.hypot(x - px, y - py) < t.pocketRadius + R) return false;
  }
  return true;
}

export function placeCue(state, x, y) {
  const cue = state.balls.find((b) => b.number === 0);
  cue.pocketed = false;
  cue.x = x;
  cue.y = y;
  cue.vx = cue.vy = cue.wx = cue.wy = cue.wz = 0;
}

/** Nearest legal cue position to a requested one, so dragging never gets stuck. */
export function nearestLegalCue(state, x, y) {
  if (canPlaceCue(state, x, y)) return [x, y];
  const R = BALL.radius;
  for (let ring = 1; ring <= 40; ring++) {
    const radius = ring * R * 0.4;
    for (let k = 0; k < 24; k++) {
      const a = (k / 24) * Math.PI * 2;
      const cx = x + radius * Math.cos(a);
      const cy = y + radius * Math.sin(a);
      if (canPlaceCue(state, cx, cy)) return [cx, cy];
    }
  }
  const [hx, hy] = headSpot(state.table);
  return [hx, hy];
}

export function ensureCueOnTable(state) {
  const cue = state.balls.find((b) => b.number === 0);
  if (!cue || (!cue.pocketed && cue.x > 0)) return;
  const [hx, hy] = headSpot(state.table);
  const [x, y] = nearestLegalCue(state, hx, hy);
  if (!cue) state.balls.unshift(makeBall(0, x, y));
  else placeCue(state, x, y);
}

export function scoreboard(state) {
  const counts = { solid: 0, stripe: 0 };
  for (const b of state.balls) {
    if (b.pocketed || b.number === 0 || b.number === 8) continue;
    counts[suitOf(b.number)]++;
  }
  return counts;
}
