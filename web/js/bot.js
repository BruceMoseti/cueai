/**
 * The opponent.
 *
 * The bot is built around the finding that motivates this repository: aiming a
 * pot needs no model at all, because the ghost-ball construction is exact, so
 * the interesting problem is *choosing* which of the exact shots to play. That
 * is a search, and a search is worth exactly as much as the number of rollouts
 * you can afford.
 *
 *   1. Enumerate every (ball, pocket) pair and solve the aim in closed form.
 *      Microseconds each, so this prunes the space for free.
 *   2. Throw away anything blocked or cut too thin to make.
 *   3. Simulate what survives, with the same physics the game itself runs, and
 *      score the resting layout rather than just whether the ball dropped.
 *   4. Add the aiming error a human wrist has, so the plan and the outcome are
 *      allowed to differ.
 *
 * Step 3 is the part that costs something, and it is why the bot's strength is
 * reported in rollouts rather than in adjectives.
 */

import { applyShot, simulateToRest } from "./physics.js";
import { cloneBalls, suitOf } from "./rack.js";
import { ghostBall, pathClear } from "./aim.js";
import {
  canPlaceCue as canPlaceCueAt,
  groupCleared,
  legalTargets,
  opponentOf,
  resolveShot,
} from "./game.js";

export const DIFFICULTIES = {
  relaxed: { label: "Relaxed", aimErrorDeg: 1.1, speeds: [2.0, 3.4], maxCandidates: 6, lookahead: false },
  club: { label: "Club player", aimErrorDeg: 0.45, speeds: [1.6, 2.6, 3.8], maxCandidates: 10, lookahead: true },
  sharp: { label: "Sharp", aimErrorDeg: 0.12, speeds: [1.4, 2.2, 3.2, 4.4], maxCandidates: 14, lookahead: true },
};

const MAX_CUT = (78 * Math.PI) / 180;
// The search runs at a coarser step than the shot it is planning. Four
// milliseconds keeps a rollout honest about pots and scratches while costing a
// quarter as much, which buys four times the candidates.
const SEARCH_DT = 0.004;
const SEARCH_MAX_TIME = 9.0;

function yieldToUI() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function trialState(state) {
  return { ...state, balls: cloneBalls(state.balls), groups: { ...state.groups } };
}

/**
 * Enumerate potting geometry. Everything here is closed form.
 */
function candidateShots(state, player) {
  const cue = state.balls.find((b) => b.number === 0);
  const targets = legalTargets(state, player);
  const candidates = [];

  for (const target of targets) {
    for (const pocket of state.table.pockets) {
      const g = ghostBall(cue, target, pocket);
      if (!g || g.cut > MAX_CUT) continue;
      if (!pathClear(state.balls, cue.x, cue.y, g.x, g.y, [0, target.number])) continue;
      if (!pathClear(state.balls, target.x, target.y, pocket[0], pocket[1], [0, target.number])) continue;

      // A rough pottability prior, used only to decide what is worth
      // simulating. Thin cuts and long object travel are what miss.
      const difficulty = (1 / Math.cos(g.cut)) * (1 + g.objTravel) * (1 + 0.35 * g.cueTravel);
      candidates.push({ kind: "pot", target: target.number, pocket, angle: g.angle, cut: g.cut, difficulty, geometry: g });
    }
  }

  candidates.sort((a, b) => a.difficulty - b.difficulty);
  return candidates;
}

/** When nothing can be potted, roll up behind something instead of blasting. */
function safetyShots(state, player) {
  const cue = state.balls.find((b) => b.number === 0);
  const targets = legalTargets(state, player);
  const shots = [];
  for (const target of targets) {
    const angle = Math.atan2(target.y - cue.y, target.x - cue.x);
    for (const speed of [1.1, 1.9]) {
      shots.push({ kind: "safety", target: target.number, angle, speed, difficulty: 0 });
    }
  }
  return shots.slice(0, 8);
}

/**
 * How much I like the table I would be left with.
 *
 * Purely geometric, so it costs nothing on top of the rollout: is there a pot
 * available, and how comfortable is it.
 */
function positionValue(state, player) {
  const cue = state.balls.find((b) => b.number === 0);
  if (!cue || cue.pocketed) return 0;
  let best = 0;
  for (const target of legalTargets(state, player)) {
    for (const pocket of state.table.pockets) {
      const g = ghostBall(cue, target, pocket);
      if (!g || g.cut > MAX_CUT) continue;
      if (!pathClear(state.balls, cue.x, cue.y, g.x, g.y, [0, target.number])) continue;
      if (!pathClear(state.balls, target.x, target.y, pocket[0], pocket[1], [0, target.number])) continue;
      // Straight and short is worth more than thin and long.
      const value = Math.cos(g.cut) / (1 + 0.4 * g.objTravel + 0.15 * g.cueTravel);
      if (value > best) best = value;
    }
  }
  return best;
}

function scoreOutcome(state, before, outcome, player) {
  const group = before.groups[player];
  let score = 0;

  for (const n of outcome.potted) {
    if (n === 0) continue; // the foul penalty below already covers the scratch
    if (n === 8) continue; // handled through the game result
    const mine = group ? suitOf(n) === group : outcome.assigned ? suitOf(n) === outcome.assigned : true;
    score += mine ? 1000 : -750;
  }

  if (state.phase === "over") {
    score += state.winner === player ? 100000 : -100000;
  }

  if (outcome.foul) score -= 1400;
  if (outcome.continues) score += 300;

  // What the shot leaves behind, for me if I keep shooting, against me if not.
  const nextPlayer = outcome.continues ? player : opponentOf(player);
  const value = positionValue(state, nextPlayer);
  score += (outcome.continues ? 500 : -420) * value;

  return score;
}

function evaluate(state, player, shot, stats) {
  const trial = trialState(state);
  trial.turn = player;
  const context = {
    wasBreak: trial.phase === "break",
    clearedBefore: groupCleared(trial, player),
  };
  const cue = trial.balls.find((b) => b.number === 0);
  applyShot(cue, shot);
  const events = simulateToRest(trial.balls, trial.table, {
    dt: SEARCH_DT,
    maxTime: SEARCH_MAX_TIME,
  });
  stats.rollouts++;
  stats.tableSeconds += events.tableTime;
  const outcome = resolveShot(trial, events, context);
  return { score: scoreOutcome(trial, state, outcome, player), outcome, events };
}

/**
 * Where to put the cue ball with ball in hand.
 *
 * Ball in hand is the largest advantage in the game and spending it on a
 * random legal square wastes it, so this scores a grid of placements with the
 * same geometric position value the shot search uses.
 */
export function choosePlacement(state, player = state.turn) {
  const t = state.table;
  const xMax = state.behindHeadString ? t.length * 0.25 : t.length;
  let best = null;
  const trial = trialState(state);
  const cue = trial.balls.find((b) => b.number === 0);
  cue.pocketed = false;

  for (let i = 1; i < 26; i++) {
    for (let j = 1; j < 14; j++) {
      const x = (xMax * i) / 26;
      const y = (t.width * j) / 14;
      if (!canPlaceCueAt(state, x, y)) continue;
      cue.x = x;
      cue.y = y;
      const value = positionValue(trial, player);
      if (!best || value > best.value) best = { x, y, value };
    }
  }
  return best ?? { x: t.length * 0.25, y: t.width * 0.5, value: 0 };
}

function breakShot(state) {
  const cue = state.balls.find((b) => b.number === 0);
  const apex = state.balls
    .filter((b) => !b.pocketed && b.number !== 0)
    .reduce((a, b) => (b.x < a.x ? b : a));
  // Slightly off dead centre: a perfectly square break sends the energy
  // straight back down the table instead of into the corners.
  const angle = Math.atan2(apex.y - cue.y, apex.x - cue.x) + 0.004;
  return { speed: 8.2, angle, englishX: 0, englishY: 0.1 };
}

/**
 * Pick a shot. Yields to the event loop between candidates so the page stays
 * responsive and can show what the search is doing.
 */
export async function chooseShot(state, { difficulty = "club", onProgress = null, rng = Math.random } = {}) {
  const settings = DIFFICULTIES[difficulty] ?? DIFFICULTIES.club;
  const player = state.turn;
  const started = performance.now();
  const stats = { rollouts: 0, tableSeconds: 0, candidates: 0, pruned: 0 };

  if (state.phase === "break") {
    const shot = breakShot(state);
    return {
      shot,
      plan: { kind: "break", note: "opening break" },
      stats: { ...stats, elapsedMs: performance.now() - started },
      considered: [],
    };
  }

  const pots = candidateShots(state, player);
  const kept = pots.slice(0, settings.maxCandidates);
  stats.candidates = pots.length;
  stats.pruned = pots.length - kept.length;

  const trials = [];
  for (const candidate of kept) {
    for (const speed of settings.speeds) {
      trials.push({ ...candidate, speed, englishX: 0, englishY: 0 });
    }
  }
  if (trials.length === 0) {
    for (const safety of safetyShots(state, player)) {
      trials.push({ ...safety, englishX: 0, englishY: 0 });
    }
  }

  let best = null;
  const considered = [];
  for (let i = 0; i < trials.length; i++) {
    const trial = trials[i];
    const shot = {
      speed: trial.speed,
      angle: trial.angle,
      englishX: trial.englishX,
      englishY: trial.englishY,
    };
    const result = evaluate(state, player, shot, stats);
    considered.push({
      target: trial.target,
      kind: trial.kind,
      speed: trial.speed,
      score: result.score,
      potted: result.outcome.potted,
      foul: result.outcome.foul,
    });
    if (!best || result.score > best.score) best = { ...result, shot, trial };
    if (onProgress && i % 2 === 0) onProgress(i + 1, trials.length);
    if (i % 2 === 1) await yieldToUI();
  }

  if (!best) {
    // No legal target is reachable at all; roll gently and concede the foul.
    return {
      shot: { speed: 1.0, angle: rng() * Math.PI * 2, englishX: 0, englishY: 0 },
      plan: { kind: "hopeless", note: "no legal target could be reached" },
      stats: { ...stats, elapsedMs: performance.now() - started },
      considered,
    };
  }

  // Execution error. The planned line is exact; the stroke is not. The repo
  // measures the potting window at a fraction of a degree, so this is the
  // single number that decides how often the bot actually makes the ball.
  const sigma = (settings.aimErrorDeg * Math.PI) / 180;
  const gauss = Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
  const aimError = sigma * gauss;
  const played = { ...best.shot, angle: best.shot.angle + aimError };

  considered.sort((a, b) => b.score - a.score);
  return {
    shot: played,
    plan: {
      kind: best.trial.kind,
      target: best.trial.target,
      pocket: best.trial.pocket,
      cut: best.trial.cut,
      score: best.score,
      aimErrorDeg: (aimError * 180) / Math.PI,
      predicted: best.outcome.potted,
    },
    stats: { ...stats, elapsedMs: performance.now() - started },
    considered: considered.slice(0, 6),
  };
}
