/**
 * Play the bot against itself, headless.
 *
 * The rules engine has more branches than a rendering bug will ever reveal:
 * open tables, ball in hand, fouling on the eight, running out of legal
 * targets. Self-play walks all of them thousands of times and fails if a game
 * fails to terminate, a ball ends up off the table, or the referee reaches a
 * state it cannot describe. It also prints how the difficulty settings
 * actually play, because "sharp" is only a useful label if it wins.
 *
 *   node web/test/selfplay.mjs [--games 20] [--difficulty club] [--verbose]
 */

import { applyShot, simulateToRest, BALL } from "../js/physics.js";
import { createGame, groupCleared, resolveShot, placeCue, YOU, BOT } from "../js/game.js";
import { chooseShot, choosePlacement, DIFFICULTIES } from "../js/bot.js";
import { mulberry32 } from "../js/rack.js";

const GAME_DT = 0.001;
const MAX_SHOTS = 250;
// Balls are separated by projection rather than by solving for the exact
// contact time, so some overlap is inherent and the useful question is whether
// it is visible. On a 2.54 m table drawn a thousand pixels wide a pixel is
// 2.5 mm, so this is the threshold at which two balls would start to look like
// they were sharing space. Twenty games measure a worst case around 0.5 mm.
const MAX_OVERLAP = 0.002;
// How many of the fifteen a full break has to actually move. Well under what a
// square hit achieves, and well above what clipping the apex leaves behind, so
// it fails on a broken break rather than on an unlucky one.
const MIN_BROKEN = 10;

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, fallback) => {
    const i = args.indexOf(flag);
    return i >= 0 ? args[i + 1] : fallback;
  };
  const difficulty = get("--difficulty", "club");
  return {
    games: Number(get("--games", 12)),
    difficulty,
    // Setting both sides differently checks that the difficulty labels order
    // the way they claim to, which is the only thing that makes them useful.
    opponent: get("--vs", difficulty),
    verbose: args.includes("--verbose"),
  };
}

/**
 * Checked while the balls are still moving, not only where they stop.
 *
 * Two balls sharing space and a ball inside a cushion are the two failures a
 * viewer would notice immediately and no resting-position check can see, since
 * both are resolved before the shot ends.
 */
function checkMidFlight(balls, table, t, worst) {
  const R = BALL.radius;
  for (let i = 0; i < balls.length; i++) {
    const a = balls[i];
    if (a.pocketed) continue;
    const out = Math.max(R - a.x, a.x - (table.length - R), R - a.y, a.y - (table.width - R));
    if (out > MAX_OVERLAP) {
      throw new Error(
        `ball ${a.number} was ${(out * 1000).toFixed(2)} mm inside a cushion at t=${t.toFixed(3)} s`
      );
    }
    for (let j = i + 1; j < balls.length; j++) {
      const b = balls[j];
      if (b.pocketed) continue;
      const overlap = 2 * R - Math.hypot(b.x - a.x, b.y - a.y);
      if (overlap > worst.overlap) worst.overlap = overlap;
      if (overlap > MAX_OVERLAP) {
        throw new Error(
          `balls ${a.number} and ${b.number} overlapped by ` +
            `${(overlap * 1000).toFixed(2)} mm at t=${t.toFixed(3)} s`
        );
      }
    }
  }
}

function checkInvariants(state, shotNumber) {
  const t = state.table;
  for (const b of state.balls) {
    if (b.pocketed) continue;
    if (!Number.isFinite(b.x) || !Number.isFinite(b.y)) {
      throw new Error(`ball ${b.number} has non-finite position after shot ${shotNumber}`);
    }
    const R = BALL.radius;
    if (b.x < R - 1e-6 || b.x > t.length - R + 1e-6 || b.y < R - 1e-6 || b.y > t.width - R + 1e-6) {
      throw new Error(
        `ball ${b.number} escaped the table at (${b.x.toFixed(4)}, ${b.y.toFixed(4)}) after shot ${shotNumber}`
      );
    }
  }
  const groups = state.groups;
  if (groups[YOU] && groups[BOT] && groups[YOU] === groups[BOT]) {
    throw new Error("both players were assigned the same group");
  }
  const cue = state.balls.find((b) => b.number === 0);
  if (cue.pocketed && !state.ballInHand && state.phase !== "over") {
    throw new Error(`the cue ball is off the table with no ball in hand after shot ${shotNumber}`);
  }
}

/**
 * Check that the break actually opened the rack.
 *
 * A break that clips the apex instead of striking it leaves the triangle
 * standing, and nothing else here would notice: the shot is legal, the game
 * continues, the invariants hold. It looks exactly like a physics limitation,
 * which is what makes it worth asserting rather than eyeballing. Striking the
 * apex square moves twelve of the fifteen; clipping it moves five.
 */
function breakOpened(state, before, seed, worst) {
  const balls = state.balls.filter((b) => b.number !== 0);
  const moved = balls.filter(
    (b, i) => b.pocketed || Math.hypot(b.x - before[i][0], b.y - before[i][1]) > 0.05
  ).length;
  if (moved < worst.leastMoved) worst.leastMoved = moved;
  worst.breaks++;
  worst.movedTotal += moved;
  if (moved < MIN_BROKEN) {
    throw new Error(
      `the break in game ${seed} moved only ${moved} of ${balls.length} balls ` +
        `more than 5 cm; the rack did not open`
    );
  }
}

async function playGame(seed, difficulties, verbose, worst) {
  const state = createGame(seed);
  const rng = mulberry32(seed ^ 0x9e3779b9);
  let shots = 0;
  let fouls = 0;
  let potted = 0;

  while (state.phase !== "over" && shots < MAX_SHOTS) {
    if (state.ballInHand || state.phase === "break") {
      const spot = choosePlacement(state);
      placeCue(state, spot.x, spot.y);
      state.ballInHand = false;
    }

    const context = {
      wasBreak: state.phase === "break",
      clearedBefore: groupCleared(state, state.turn),
    };
    const shooter = state.turn;
    const decision = await chooseShot(state, { difficulty: difficulties[shooter], rng });
    const cue = state.balls.find((b) => b.number === 0);
    const rackBefore = context.wasBreak
      ? state.balls.filter((b) => b.number !== 0).map((b) => [b.x, b.y])
      : null;
    applyShot(cue, decision.shot);
    const events = simulateToRest(state.balls, state.table, {
      dt: GAME_DT,
      maxTime: 15,
      onStep: (balls, t) => checkMidFlight(balls, state.table, t, worst),
    });
    const outcome = resolveShot(state, events, context);

    shots++;
    if (outcome.foul) fouls++;
    potted += outcome.objectPotted.length;
    checkInvariants(state, shots);
    if (rackBefore) breakOpened(state, rackBefore, seed, worst);

    if (verbose) {
      const who = context.wasBreak ? "break" : shooter;
      console.log(
        `  ${String(shots).padStart(3)} ${who.padEnd(5)} ` +
          `plan=${decision.plan.kind}:${decision.plan.target ?? "-"} ` +
          `potted=[${outcome.potted}] ${outcome.foul ? "FOUL " + outcome.fouls[0] : ""}`
      );
    }
  }

  if (state.phase !== "over") {
    throw new Error(`game ${seed} did not finish in ${MAX_SHOTS} shots`);
  }
  return { winner: state.winner, shots, fouls, potted, reason: state.loseReason };
}

async function main() {
  const { games, difficulty, opponent, verbose } = parseArgs();
  for (const name of [difficulty, opponent]) {
    if (!DIFFICULTIES[name]) {
      console.error(`unknown difficulty "${name}"; expected one of ${Object.keys(DIFFICULTIES)}`);
      process.exit(2);
    }
  }
  const difficulties = { [YOU]: difficulty, [BOT]: opponent };

  const started = Date.now();
  const results = [];
  const worst = { overlap: 0, leastMoved: Infinity, breaks: 0, movedTotal: 0 };
  for (let i = 0; i < games; i++) {
    if (verbose) console.log(`game ${i}`);
    results.push(await playGame(1000 + i, difficulties, verbose, worst));
  }

  const elapsed = (Date.now() - started) / 1000;
  const shots = results.reduce((a, r) => a + r.shots, 0);
  const fouls = results.reduce((a, r) => a + r.fouls, 0);
  const potted = results.reduce((a, r) => a + r.potted, 0);
  const wins = results.filter((r) => r.winner === YOU).length;

  console.log(`\n${games} games, "${difficulty}" against "${opponent}", in ${elapsed.toFixed(1)}s`);
  console.log(`  ${(shots / games).toFixed(1)} shots per game, ${(elapsed / shots).toFixed(2)}s per shot`);
  console.log(`  ${((potted / shots) * 100).toFixed(0)}% of shots potted a ball`);
  console.log(`  ${((fouls / shots) * 100).toFixed(0)}% of shots fouled`);
  console.log(`  "${difficulty}" won ${wins}/${games}`);
  console.log(
    `  worst ball overlap while moving ${(worst.overlap * 1000).toFixed(3)} mm ` +
      `on a ${(2 * BALL.radius * 1000).toFixed(1)} mm ball`
  );
  console.log(
    `  the break moved ${(worst.movedTotal / worst.breaks).toFixed(1)}/15 balls on average, ` +
      `${worst.leastMoved} at worst`
  );
  console.log("  every game reached a legal conclusion");
}

main().catch((error) => {
  console.error(`\nself-play failed: ${error.message}`);
  process.exit(1);
});
