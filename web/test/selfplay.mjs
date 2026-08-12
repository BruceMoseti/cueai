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

async function playGame(seed, difficulties, verbose) {
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
    applyShot(cue, decision.shot);
    const events = simulateToRest(state.balls, state.table, { dt: GAME_DT, maxTime: 15 });
    const outcome = resolveShot(state, events, context);

    shots++;
    if (outcome.foul) fouls++;
    potted += outcome.objectPotted.length;
    checkInvariants(state, shots);

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
  for (let i = 0; i < games; i++) {
    if (verbose) console.log(`game ${i}`);
    results.push(await playGame(1000 + i, difficulties, verbose));
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
  console.log("  every game reached a legal conclusion");
}

main().catch((error) => {
  console.error(`\nself-play failed: ${error.message}`);
  process.exit(1);
});
