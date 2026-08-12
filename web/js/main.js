/**
 * Wiring: input, the animation loop, and whose turn it is.
 *
 * Shots are stepped in real time at the reference timestep rather than
 * simulated up front and replayed, so what the inspector reports is the state
 * of the ball on screen at that instant.
 */

import { anyBallMoving, applyShot, newEvents, stepWorld, MAX_TIP_OFFSET } from "./physics.js";
import {
  BOT,
  YOU,
  canPlaceCue,
  createGame,
  groupCleared,
  legalTargets,
  nearestLegalCue,
  placeCue,
  resolveShot,
} from "./game.js";
import { chooseShot, choosePlacement, DIFFICULTIES } from "./bot.js";
import { Renderer, drawSpinWidget } from "./render.js";
import { Inspector } from "./inspector.js";
import { firstContact } from "./aim.js";
import { BALL_COLORS, suitOf } from "./rack.js";
import { loadFacts } from "./facts.js";

const PHYS_DT = 0.001;
const MAX_CUE_SPEED = 7.5; // m/s at full power, a hard break
// Enough headroom for 3x playback on a 30 Hz display; past that the frame is
// late anyway and catching up fully would make it later.
const MAX_SUBSTEPS = 170;
// Every shot now reaches rest well inside this, but a turn that never ends is
// an unrecoverable game rather than a visible glitch, so the loop is bounded.
const MAX_SHOT_SECONDS = 20;
const STROKE_MS = 200; // backswing and delivery, before the ball is struck
const DROP_MS = 260; // a potted ball falling out of sight
const DRAG_DEADZONE = 0.012; // metres of pull-back that count as drawing the cue
const CLICK_HOLD_MS = 220; // a press held this long on the spot is a shot, not a nudge

const el = (id) => document.getElementById(id);
const ui = {
  canvas: el("table"),
  spin: el("spin"),
  power: el("power"),
  powerReadout: el("power-readout"),
  shoot: el("shoot"),
  newgame: el("newgame"),
  turnLabel: el("turn-label"),
  turnDot: el("turn-dot"),
  message: el("message"),
  groupYou: el("group-you"),
  groupBot: el("group-bot"),
  ballsYou: el("balls-you"),
  ballsBot: el("balls-bot"),
  cardYou: el("card-you"),
  cardBot: el("card-bot"),
  difficulty: el("difficulty"),
  playback: el("playback"),
  showAim: el("show-aim"),
  showTargets: el("show-targets"),
  botProgress: el("bot-progress"),
  botReport: el("bot-report"),
  shotlog: el("shotlog-list"),
  banner: el("banner"),
  bannerTitle: el("banner-title"),
  bannerText: el("banner-text"),
  bannerButton: el("banner-button"),
};

const renderer = new Renderer(ui.canvas);
const inspector = new Inspector({
  chip: el("state-chip"),
  stats: el("inspector-stats"),
  canvas: el("trace"),
  legend: el("trace-legend"),
});

let state;
let mode; // "placing" | "aim" | "stroking" | "rolling" | "thinking" | "over"
let aimAngle = 0;
let power = 0.42;
let spin = { x: 0, y: 0 };
let pointerTable = null;
let drag = null;
let shotEvents = null;
let shotContext = null;
let tableTime = 0;
let lastFrame = performance.now();
let messageHtml = "";
let history = [];
// Table time owed from the previous frame. Rounding the substep count instead
// would make the playback speed a function of the frame rate, which shows up
// as the balls surging whenever the browser misses a frame.
let physDebt = 0;
let stroke = null; // { shot, startedAt } while the cue is being delivered
let drops = []; // potted balls, still falling for the eye's benefit
let botAimPreview = false; // show the bot's line between deciding and striking
// The bot's turn spans several awaits. Starting a new game in the middle of
// one has to invalidate it, or the old turn wakes up and shoots on the new
// table.
let generation = 0;

// ---------- lifecycle ----------

function newGame() {
  generation++;
  state = createGame();
  mode = "placing";
  aimAngle = 0;
  spin = { x: 0, y: 0 };
  shotEvents = null;
  tableTime = 0;
  history = [];
  physDebt = 0;
  stroke = null;
  drops = [];
  botAimPreview = false;
  inspector.reset();
  messageHtml = "Ball in hand behind the head string. Click to place the cue ball, then break.";
  ui.banner.classList.remove("show");
  ui.botReport.textContent =
    "The bot solves every pot in closed form, then simulates the ones worth considering.";
  ui.botProgress.style.width = "0%";
  drawSpinWidget(ui.spin, spin);
  syncUI();
}

function cueBall() {
  return state.balls.find((b) => b.number === 0);
}

// ---------- shooting ----------

/**
 * Draw the cue back and deliver it, then hand over to the physics.
 *
 * The stroke is animation only — the shot handed to the simulator is the one
 * that was chosen before it started — but a ball that leaps off a stationary
 * cue reads as a state change rather than as a stroke.
 */
function beginStroke(shot) {
  const cue = cueBall();
  if (!cue || cue.pocketed) return;
  stroke = { shot, startedAt: performance.now() };
  mode = "stroking";
  syncUI();
}

/** Backswing, then accelerate into the ball. 1 is at rest, 0 is contact. */
function strokeOffset(p) {
  const BACKSWING = 0.3;
  if (p < 0.36) return 1 + BACKSWING * (p / 0.36);
  const q = (p - 0.36) / 0.64;
  return (1 + BACKSWING) * (1 - q * q);
}

function beginShot(shot) {
  const cue = cueBall();
  if (!cue || cue.pocketed) return;
  shotContext = {
    wasBreak: state.phase === "break",
    clearedBefore: groupCleared(state, state.turn),
    shooter: state.turn,
  };
  shotEvents = newEvents();
  tableTime = 0;
  physDebt = 0;
  botAimPreview = false;
  inspector.beginShot(shot.speed);
  applyShot(cue, shot);
  mode = "rolling";
  syncUI();
}

function playerShoot() {
  if (mode !== "aim") return;
  beginStroke({
    speed: Math.max(0.35, power * MAX_CUE_SPEED),
    angle: aimAngle,
    englishX: spin.x,
    englishY: spin.y,
  });
}

function finishShot() {
  const outcome = resolveShot(state, shotEvents, shotContext);
  messageHtml = describeOutcome(outcome, shotContext);
  history.push({
    shooter: shotContext.shooter,
    wasBreak: shotContext.wasBreak,
    potted: [...outcome.potted],
    foul: outcome.foul,
    fouls: [...outcome.fouls],
    continues: outcome.continues,
    cushions: shotEvents.cushions,
    firstContact: shotEvents.firstContact,
  });

  if (state.phase === "over") {
    mode = "over";
    showBanner();
    syncUI();
    return;
  }

  if (state.turn === BOT) {
    startBotTurn();
  } else {
    mode = state.ballInHand ? "placing" : "aim";
    aimAtNearestTarget();
  }
  syncUI();
}

function describeOutcome(outcome, context) {
  const bits = [];
  if (context.wasBreak) bits.push("Break.");
  if (outcome.potted.length) {
    const objects = outcome.objectPotted;
    if (objects.length) bits.push(`Potted ${objects.map((n) => `the ${n}`).join(", ")}.`);
  } else if (!outcome.foul) {
    bits.push("No pot.");
  }
  if (outcome.assigned) {
    const mine = state.groups[YOU] === outcome.assigned;
    bits.push(`Groups set: you are ${state.groups[YOU]}s${mine ? "" : ""}.`);
  }
  if (outcome.foul) {
    bits.push(`<span class="foul">Foul — ${outcome.fouls[0]}.</span>`);
    bits.push(`${state.turn === YOU ? "You have" : "The bot has"} ball in hand.`);
  } else if (outcome.continues) {
    bits.push(state.turn === YOU ? "Shoot again." : "The bot keeps the table.");
  }
  return bits.join(" ");
}

function showBanner() {
  const won = state.winner === YOU;
  ui.bannerTitle.textContent = won ? "You win" : "The bot wins";
  ui.bannerTitle.style.color = won ? "var(--good)" : "var(--warn)";
  ui.bannerText.textContent = state.loseReason
    ? `The 8 decided it: ${state.loseReason}.`
    : won
      ? "Group cleared and the 8 dropped on a legal shot."
      : "The bot cleared its group and made the 8.";
  ui.banner.classList.add("show");
}

// ---------- the opponent ----------

async function startBotTurn() {
  const era = generation;
  const stale = () => era !== generation;

  mode = "thinking";
  syncUI();
  await pause(260); // a beat, so the change of turn reads
  if (stale()) return;

  if (state.ballInHand) {
    const spot = choosePlacement(state);
    placeCue(state, spot.x, spot.y);
    state.ballInHand = false;
    syncUI();
    await pause(220);
    if (stale()) return;
  }

  let decision;
  try {
    decision = await chooseShot(state, {
      difficulty: ui.difficulty.value,
      onProgress: (done, total) => {
        if (!stale()) ui.botProgress.style.width = `${Math.round((done / total) * 100)}%`;
      },
    });
  } catch (error) {
    // Never strand the turn: concede rather than leave the table frozen.
    console.error("the bot failed to choose a shot", error);
    decision = { shot: { speed: 1.2, angle: aimAngle, englishX: 0, englishY: 0 }, plan: { kind: "hopeless" }, stats: {} };
  }
  if (stale()) return;

  ui.botProgress.style.width = "100%";
  ui.botReport.innerHTML = reportBotDecision(decision);
  aimAngle = decision.shot.angle;
  botAimPreview = true;
  await pause(420); // let the chosen line be seen before the cue moves
  if (stale()) {
    botAimPreview = false;
    return;
  }

  ui.botProgress.style.width = "0%";
  beginStroke(decision.shot);
}

function pause(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function reportBotDecision(decision) {
  const { plan, stats } = decision;
  if (plan.kind === "break") {
    return "<b>Break.</b> No search: the opening shot has nothing to choose between.";
  }

  const lines = [];
  if (stats.rollouts) {
    // What the search spent, in its own units. The table time is the number
    // worth reading: it is how much simulated billiards fitted in the pause.
    const cost =
      `${stats.tableSeconds.toFixed(1)} s of table time ` +
      `in ${Math.round(stats.elapsedMs)} ms`;
    if (stats.candidates) {
      const pruned = stats.pruned ? `, ${stats.pruned} pruned` : "";
      lines.push(
        `<b>${stats.candidates}</b> pot line${stats.candidates === 1 ? "" : "s"} ` +
          `solved in closed form${pruned}, <b>${stats.rollouts}</b> simulated (${cost}).`
      );
    } else {
      lines.push(
        `No pot line exists from here. <b>${stats.rollouts}</b> safety shots ` +
          `simulated (${cost}).`
      );
    }
  }
  if (plan.kind === "pot") {
    lines.push(
      `Chose the <b>${plan.target}</b>, cut <b>${((plan.cut * 180) / Math.PI).toFixed(0)}°</b>.`
    );
  } else if (plan.kind === "safety") {
    lines.push(`Playing safe off the <b>${plan.target}</b>.`);
  } else {
    lines.push("No legal target could be reached from here.");
  }
  if (Number.isFinite(plan.aimErrorDeg)) {
    lines.push(
      `Stroke error <b>${plan.aimErrorDeg >= 0 ? "+" : ""}${plan.aimErrorDeg.toFixed(2)}°</b> ` +
        `(${DIFFICULTIES[ui.difficulty.value].label}).`
    );
  }
  return lines.join("<br>");
}

// ---------- aiming helpers ----------

function aimAtNearestTarget() {
  const cue = cueBall();
  if (!cue || cue.pocketed) return;
  const targets = legalTargets(state, YOU);
  if (!targets.length) return;
  const nearest = targets.reduce((a, b) =>
    Math.hypot(b.x - cue.x, b.y - cue.y) < Math.hypot(a.x - cue.x, a.y - cue.y) ? b : a
  );
  aimAngle = Math.atan2(nearest.y - cue.y, nearest.x - cue.x);
}

function currentAim() {
  const cue = cueBall();
  if (!cue || cue.pocketed) return null;
  return firstContact(state.balls, cue, aimAngle, state.table);
}

// ---------- input ----------

function pointerPosition(event) {
  const rect = ui.canvas.getBoundingClientRect();
  return renderer.toTable(event.clientX - rect.left, event.clientY - rect.top);
}

ui.canvas.addEventListener("pointermove", (event) => {
  pointerTable = pointerPosition(event);
  if (mode === "aim" && !drag) {
    const cue = cueBall();
    if (!cue || cue.pocketed) return;
    const target = Math.atan2(pointerTable[1] - cue.y, pointerTable[0] - cue.x);
    if (event.shiftKey) {
      // Ease toward the pointer so the last fraction of a degree is reachable
      // with a mouse; the potting window is narrower than one pixel of arc.
      let delta = target - aimAngle;
      while (delta > Math.PI) delta -= 2 * Math.PI;
      while (delta < -Math.PI) delta += 2 * Math.PI;
      aimAngle += delta * 0.08;
    } else {
      aimAngle = target;
    }
  } else if (drag) {
    const back =
      -((pointerTable[0] - drag.x) * Math.cos(aimAngle) +
        (pointerTable[1] - drag.y) * Math.sin(aimAngle));
    if (back > DRAG_DEADZONE) drag.pulled = true;
    if (drag.pulled) {
      power = Math.max(0.06, Math.min(1, back / 0.42));
      ui.power.value = String(power);
      updatePowerReadout();
    }
  }
});

ui.canvas.addEventListener("pointerdown", (event) => {
  ui.canvas.setPointerCapture(event.pointerId);
  const [x, y] = pointerPosition(event);

  if (mode === "placing") {
    const [px, py] = nearestLegalCue(state, x, y);
    placeCue(state, px, py);
    state.ballInHand = false;
    mode = "aim";
    aimAtNearestTarget();
    messageHtml =
      state.phase === "break" ? "Break when you are ready." : "Cue ball placed. Take your shot.";
    syncUI();
    return;
  }

  if (mode === "aim") drag = { x, y, pulled: false, at: performance.now() };
});

ui.canvas.addEventListener("pointerup", (event) => {
  if (mode === "aim" && drag) {
    // A shot costs a turn, so it takes a deliberate gesture: draw the cue back,
    // or hold still on the ball. A quick click is someone lining the shot up.
    const deliberate = drag.pulled || performance.now() - drag.at > CLICK_HOLD_MS;
    drag = null;
    if (deliberate) playerShoot();
  }
  void event;
});

ui.canvas.addEventListener("pointercancel", () => {
  drag = null;
});

ui.canvas.addEventListener("pointerleave", () => {
  if (mode !== "placing") pointerTable = null;
});

window.addEventListener("keydown", (event) => {
  // The controls are real form elements; space and the arrows belong to
  // whichever one has focus before they belong to the table.
  const target = event.target;
  if (target instanceof HTMLElement && target.closest("input, select, button, textarea")) return;

  if (event.key === " " || event.key === "Enter") {
    event.preventDefault();
    playerShoot();
  }
  if (mode !== "aim") return;
  const fine = event.shiftKey ? 0.0002 : 0.0012;
  if (event.key === "ArrowLeft") {
    aimAngle -= fine;
    event.preventDefault();
  }
  if (event.key === "ArrowRight") {
    aimAngle += fine;
    event.preventDefault();
  }
  if (event.key === "ArrowUp" || event.key === "ArrowDown") {
    const step = event.key === "ArrowUp" ? 0.04 : -0.04;
    power = Math.max(0.06, Math.min(1, power + step));
    ui.power.value = String(power);
    updatePowerReadout();
    event.preventDefault();
  }
});

function updatePowerReadout() {
  ui.powerReadout.textContent = `${(power * MAX_CUE_SPEED).toFixed(1)} m/s`;
}

ui.power.addEventListener("input", () => {
  power = Number(ui.power.value);
  updatePowerReadout();
});

ui.shoot.addEventListener("click", playerShoot);
ui.newgame.addEventListener("click", newGame);
ui.bannerButton.addEventListener("click", newGame);

function spinFromEvent(event) {
  const rect = ui.spin.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const r = cx * 0.86;
  let x = (event.clientX - rect.left - cx) / r;
  let y = -(event.clientY - rect.top - cy) / r;
  const mag = Math.hypot(x, y);
  if (mag > MAX_TIP_OFFSET) {
    x = (x / mag) * MAX_TIP_OFFSET;
    y = (y / mag) * MAX_TIP_OFFSET;
  }
  spin = { x, y };
  drawSpinWidget(ui.spin, spin);
}

let spinDragging = false;
ui.spin.addEventListener("pointerdown", (event) => {
  spinDragging = true;
  ui.spin.setPointerCapture(event.pointerId);
  spinFromEvent(event);
});
ui.spin.addEventListener("pointermove", (event) => {
  if (spinDragging) spinFromEvent(event);
});
ui.spin.addEventListener("pointerup", () => {
  spinDragging = false;
});

ui.playback.addEventListener("change", () => {});
window.addEventListener("resize", () => renderer.resize(state.table));

// ---------- per-frame ----------

/**
 * Note balls that have just been pocketed.
 *
 * The simulator flags a ball and stops drawing it, which between two frames
 * looks like the ball being deleted rather than falling in. Recording where it
 * went lets the renderer finish the motion.
 */
function collectDrops() {
  for (const b of state.balls) {
    if (!b.pocketed) {
      b.dropped = false;
      continue;
    }
    if (b.dropped) continue;
    b.dropped = true;
    drops.push({ number: b.number, x: b.x, y: b.y, at: performance.now() });
  }
}

function advancePhysics(elapsedSeconds) {
  const scale = Number(ui.playback.value);
  physDebt += elapsedSeconds * scale;
  let substeps = Math.floor(physDebt / PHYS_DT);
  if (substeps > MAX_SUBSTEPS) {
    // A backgrounded tab hands back a delta measured in seconds. Abandoning
    // the backlog costs a jump; paying it off costs a freeze and then a jump.
    substeps = MAX_SUBSTEPS;
    physDebt = 0;
  } else {
    physDebt -= substeps * PHYS_DT;
  }

  const cue = cueBall();
  for (let i = 0; i < substeps; i++) {
    stepWorld(state.balls, state.table, PHYS_DT, shotEvents);
    tableTime += PHYS_DT;
    for (const b of state.balls) {
      if (!b.pocketed) b.visualSpin = (b.visualSpin ?? 0) + b.wz * PHYS_DT;
    }
    collectDrops();
    // Sampled inside the loop, not once a frame: the inspector thins the trace
    // by table time so what it plots does not depend on the frame rate.
    inspector.sample(cue, tableTime, shotEvents.collisions, shotEvents.cushions);
    if (!anyBallMoving(state.balls)) break;
  }

  if (tableTime > MAX_SHOT_SECONDS && anyBallMoving(state.balls)) {
    console.warn(`shot still moving after ${MAX_SHOT_SECONDS}s of table time; forcing rest`);
    for (const b of state.balls) {
      b.vx = b.vy = b.wx = b.wy = b.wz = 0;
    }
  }
  if (!anyBallMoving(state.balls)) finishShot();
}

function frame(now) {
  const elapsed = Math.min((now - lastFrame) / 1000, 0.05);
  lastFrame = now;

  let strokeGap = 1;
  if (mode === "stroking") {
    const p = (now - stroke.startedAt) / STROKE_MS;
    if (p >= 1) {
      const shot = stroke.shot;
      stroke = null;
      beginShot(shot);
    } else {
      strokeGap = strokeOffset(p);
    }
  }

  if (mode === "rolling") advancePhysics(elapsed);
  if (drops.length) drops = drops.filter((d) => now - d.at < DROP_MS);

  const cue = cueBall();
  // The bot's chosen line is worth showing during the beat before it strokes:
  // it is the search's answer, and it is the only moment you can read it.
  const aiming = mode === "aim" || mode === "stroking" || botAimPreview;
  const aim = aiming ? currentAim() : null;
  let tangentSide = 1;
  if (aim?.hit) {
    const dot = Math.cos(aimAngle) * aim.tangentDir[0] + Math.sin(aimAngle) * aim.tangentDir[1];
    tangentSide = dot >= 0 ? 1 : -1;
  }

  // Ringing every ball on an open table says nothing, so the markers only
  // appear once the legal set is genuinely narrower than what is on the cloth.
  const yourTurn = (aiming || mode === "placing") && state.turn === YOU;
  const legal = yourTurn ? legalTargets(state, YOU) : [];
  const onTable = state.balls.filter((b) => !b.pocketed && b.number !== 0).length;
  const restricted = yourTurn && legal.length > 0 && legal.length < onTable;

  let ghostCue = null;
  if (mode === "placing" && pointerTable) {
    const [gx, gy] = pointerTable;
    ghostCue = { x: gx, y: gy, legal: canPlaceCue(state, gx, gy) };
  }

  renderer.draw(state, {
    aim,
    angle: aimAngle,
    power,
    spin,
    tangentSide,
    ghostCue,
    drops,
    dropAge: (d) => (now - d.at) / DROP_MS,
    strokeGap,
    showAim: ui.showAim.checked && aiming,
    showCue: aiming,
    showTargets: ui.showTargets.checked && restricted,
    footSpot: true,
    highlight: restricted ? new Set(legal.map((b) => b.number)) : null,
  });

  inspector.update(cue, {
    tableTime,
    collisions: shotEvents?.collisions ?? 0,
    cushions: shotEvents?.cushions ?? 0,
  });
  inspector.render();

  requestAnimationFrame(frame);
}

// ---------- panel ----------

function pip(number) {
  const stripe = suitOf(number) === "stripe";
  return (
    `<span class="pip${stripe ? " stripe" : ""}" style="--pip:${BALL_COLORS[number]};` +
    `${stripe ? "" : `background:${BALL_COLORS[number]};`}` +
    `${number === 8 ? "color:#f0f0f0;" : ""}">${number}</span>`
  );
}

const LOG_ROWS = 6;

/**
 * The last few shots, as the rules engine saw them.
 *
 * A foul is a claim about what happened — which ball was struck first, whether
 * anything reached a rail afterwards — and a game that announces only the
 * verdict is asking to be taken on trust. Every shot the referee judged stays
 * on screen with the reason it was judged that way.
 */
function renderShotLog() {
  if (!history.length) {
    ui.shotlog.innerHTML =
      `<li class="empty">Nothing yet. Place the cue ball behind the head string and break.</li>`;
    return;
  }

  ui.shotlog.innerHTML = history
    .slice(-LOG_ROWS)
    .reverse()
    .map((h) => {
      const objects = h.potted.filter((n) => n !== 0);
      const what = [];
      if (h.wasBreak) what.push("break");
      if (objects.length) what.push(`potted ${objects.map((n) => `the ${n}`).join(", ")}`);
      else if (!h.wasBreak) {
        what.push(h.firstContact === null ? "hit nothing" : `hit the ${h.firstContact} first`);
      }
      if (h.cushions) what.push(`${h.cushions} rail${h.cushions === 1 ? "" : "s"}`);
      const why = h.foul ? h.fouls[0] : h.continues ? "keeps the table" : "turn passes";
      const bot = h.shooter === BOT;
      return (
        `<li><span class="who${bot ? " bot" : ""}">${bot ? "Bot" : "You"}</span>` +
        `<span class="what">${what.join(", ")}</span>` +
        `<span class="why${h.foul ? " foul" : ""}">${why}</span></li>`
      );
    })
    .join("");
}

function syncUI() {
  const thinking = mode === "thinking";
  const rolling = mode === "rolling";
  ui.shoot.disabled = mode !== "aim";
  ui.turnDot.className = `dot${state.turn === BOT ? " bot" : ""}`;

  if (mode === "over") {
    ui.turnLabel.textContent = state.winner === YOU ? "You win" : "Bot wins";
  } else if (thinking) {
    ui.turnLabel.textContent = "Bot is thinking";
  } else if (rolling) {
    ui.turnLabel.textContent = "Balls rolling";
  } else if (state.turn === YOU) {
    ui.turnLabel.textContent =
      state.phase === "break" ? "Your break" : mode === "placing" ? "Ball in hand" : "Your shot";
  } else {
    ui.turnLabel.textContent = "Bot's shot";
  }

  ui.message.innerHTML = messageHtml;
  renderShotLog();

  for (const [who, groupEl, ballsEl, cardEl] of [
    [YOU, ui.groupYou, ui.ballsYou, ui.cardYou],
    [BOT, ui.groupBot, ui.ballsBot, ui.cardBot],
  ]) {
    const group = state.groups[who];
    const remaining = state.balls.filter(
      (b) => !b.pocketed && b.number !== 0 && b.number !== 8 && (!group || suitOf(b.number) === group)
    );
    groupEl.textContent = group
      ? groupCleared(state, who)
        ? "on the 8"
        : `${group}s`
      : "open table";
    ballsEl.innerHTML = group
      ? remaining.map((b) => pip(b.number)).join("")
      : `<span style="font-size:11.5px;color:var(--ink-faint)">not yet assigned</span>`;
    cardEl.classList.toggle("active", state.turn === who && mode !== "over");
  }
}

// ---------- start ----------

// Test seam. `web/test/browser.mjs` drives a real Chrome through this rather
// than through synthetic mouse maths, so the end-to-end test exercises the
// same functions the buttons call.
window.pocket = {
  get state() {
    return state;
  },
  get mode() {
    return mode;
  },
  get history() {
    return history;
  },
  get aimAngle() {
    return aimAngle;
  },
  get power() {
    return power;
  },
  get spin() {
    return { ...spin };
  },
  /**
   * Table metres to viewport pixels. `web/test/input.mjs` needs this to put a
   * real cursor on a particular ball, which is the only way to test the
   * pointer handling rather than the functions underneath it.
   */
  toClient(x, y) {
    const rect = ui.canvas.getBoundingClientRect();
    const [px, py] = renderer.toCanvas(x, y);
    return {
      x: rect.left + (px / ui.canvas.width) * rect.width,
      y: rect.top + (py / ui.canvas.height) * rect.height,
    };
  },
  shoot: playerShoot,
  newGame,
  place(x, y) {
    const [px, py] = nearestLegalCue(state, x, y);
    placeCue(state, px, py);
    state.ballInHand = false;
    mode = "aim";
    aimAtNearestTarget();
    syncUI();
  },
  aim(angle) {
    aimAngle = angle;
  },
  setPower(value) {
    power = value;
    ui.power.value = String(value);
    updatePowerReadout();
  },
};

newGame();
renderer.resize(state.table);
updatePowerReadout();
requestAnimationFrame(frame);
void loadFacts();
