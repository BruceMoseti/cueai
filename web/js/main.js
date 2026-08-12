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
import { Inspector, trackPeak } from "./inspector.js";
import { firstContact } from "./aim.js";
import { BALL_COLORS, suitOf } from "./rack.js";

const PHYS_DT = 0.001;
const MAX_CUE_SPEED = 7.5; // m/s at full power, a hard break
const MAX_SUBSTEPS = 90; // ceiling so a stalled tab cannot spiral

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
let mode; // "placing" | "aim" | "rolling" | "thinking" | "over"
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

// ---------- lifecycle ----------

function newGame() {
  state = createGame();
  mode = "placing";
  aimAngle = 0;
  spin = { x: 0, y: 0 };
  shotEvents = null;
  tableTime = 0;
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

function beginShot(shot) {
  const cue = cueBall();
  if (!cue || cue.pocketed) return;
  shotContext = {
    wasBreak: state.phase === "break",
    clearedBefore: groupCleared(state, state.turn),
  };
  shotEvents = newEvents();
  tableTime = 0;
  inspector.beginShot(shot.speed);
  applyShot(cue, shot);
  mode = "rolling";
  syncUI();
}

function playerShoot() {
  if (mode !== "aim") return;
  beginShot({
    speed: Math.max(0.35, power * MAX_CUE_SPEED),
    angle: aimAngle,
    englishX: spin.x,
    englishY: spin.y,
  });
}

function finishShot() {
  const outcome = resolveShot(state, shotEvents, shotContext);
  messageHtml = describeOutcome(outcome, shotContext);

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
  mode = "thinking";
  syncUI();
  await new Promise((r) => setTimeout(r, 260)); // a beat, so the turn reads

  if (state.ballInHand) {
    const spot = choosePlacement(state);
    placeCue(state, spot.x, spot.y);
    state.ballInHand = false;
    syncUI();
    await new Promise((r) => setTimeout(r, 220));
  }

  const decision = await chooseShot(state, {
    difficulty: ui.difficulty.value,
    onProgress: (done, total) => {
      ui.botProgress.style.width = `${Math.round((done / total) * 100)}%`;
    },
  });

  ui.botProgress.style.width = "100%";
  ui.botReport.innerHTML = reportBotDecision(decision);
  aimAngle = decision.shot.angle;
  await new Promise((r) => setTimeout(r, 420)); // let the cue line be seen
  ui.botProgress.style.width = "0%";
  beginShot(decision.shot);
}

function reportBotDecision(decision) {
  const { plan, stats } = decision;
  if (plan.kind === "break") {
    return `<b>Break.</b> No search: the opening shot has nothing to choose between.`;
  }
  const lines = [
    `<b>${stats.candidates}</b> pots solved in closed form, ` +
      `<b>${stats.rollouts}</b> simulated (${stats.tableSeconds.toFixed(1)} s of table time ` +
      `in ${Math.round(stats.elapsedMs)} ms).`,
  ];
  if (plan.kind === "pot") {
    lines.push(
      `Chose the <b>${plan.target}</b>, cut <b>${((plan.cut * 180) / Math.PI).toFixed(0)}°</b>.`
    );
  } else if (plan.kind === "safety") {
    lines.push(`Nothing pottable. Playing safe off the <b>${plan.target}</b>.`);
  } else {
    lines.push("No legal target could be reached.");
  }
  lines.push(
    `Stroke error <b>${plan.aimErrorDeg >= 0 ? "+" : ""}${plan.aimErrorDeg.toFixed(2)}°</b> ` +
      `(${DIFFICULTIES[ui.difficulty.value].label}).`
  );
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
    const cue = cueBall();
    const back = -((pointerTable[0] - drag.x) * Math.cos(aimAngle) + (pointerTable[1] - drag.y) * Math.sin(aimAngle));
    power = Math.max(0.06, Math.min(1, back / 0.42));
    ui.power.value = String(power);
    updatePowerReadout();
    void cue;
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

  if (mode === "aim") drag = { x, y };
});

ui.canvas.addEventListener("pointerup", (event) => {
  if (mode === "aim" && drag) {
    drag = null;
    playerShoot();
  }
  void event;
});

ui.canvas.addEventListener("pointerleave", () => {
  if (mode !== "placing") pointerTable = null;
});

window.addEventListener("keydown", (event) => {
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

function advancePhysics(elapsedSeconds) {
  const scale = Number(ui.playback.value);
  const substeps = Math.min(MAX_SUBSTEPS, Math.round((elapsedSeconds * scale) / PHYS_DT));
  const cue = cueBall();
  for (let i = 0; i < substeps; i++) {
    stepWorld(state.balls, state.table, PHYS_DT, shotEvents);
    tableTime += PHYS_DT;
    for (const b of state.balls) {
      if (!b.pocketed) b.visualSpin = (b.visualSpin ?? 0) + b.wz * PHYS_DT;
    }
    if (!anyBallMoving(state.balls)) break;
  }
  trackPeak(inspector, cue);
  inspector.sample(cue, tableTime);
  if (!anyBallMoving(state.balls)) finishShot();
}

function frame(now) {
  const elapsed = Math.min((now - lastFrame) / 1000, 0.05);
  lastFrame = now;

  if (mode === "rolling") advancePhysics(elapsed);

  const cue = cueBall();
  const aim = mode === "aim" ? currentAim() : null;
  let tangentSide = 1;
  if (aim?.hit) {
    const dot = Math.cos(aimAngle) * aim.tangentDir[0] + Math.sin(aimAngle) * aim.tangentDir[1];
    tangentSide = dot >= 0 ? 1 : -1;
  }

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
    showAim: ui.showAim.checked && mode === "aim",
    showCue: mode === "aim",
    showTargets: ui.showTargets.checked,
    footSpot: true,
    highlight:
      mode === "aim" || mode === "placing"
        ? new Set(legalTargets(state, YOU).map((b) => b.number))
        : null,
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

newGame();
renderer.resize(state.table);
updatePowerReadout();
requestAnimationFrame(frame);
