/**
 * Live readout of the cue ball's state.
 *
 * The cloth model has four regimes and the interesting one is the handover: a
 * struck ball slides, friction at the contact point spins it up 3.5 times
 * faster than it slows the centre of mass, and when the slip reaches zero the
 * ball is rolling at exactly 5/7 of its launch speed. That number is a
 * prediction, not a parameter, so plotting the measured speed against it while
 * the ball is actually moving is the shortest honest demonstration that the
 * simulation is doing mechanics rather than easing curves.
 */

import { BALL, motionState, slipVelocity, speed } from "./physics.js";

const ROLL_FRACTION = 5 / 7;
// One sample per 4 ms of table time, not per animation frame. The slide lasts
// roughly 150 ms, so a per-frame trace would describe the handover with five
// points at 60 Hz and fewer than that at brisk playback — the resolution of
// the thing this panel exists to show would depend on the monitor.
const SAMPLE_DT = 0.004;
// The cue ball stopping does not end the shot, but plotting the flat line
// while the rest of the table finishes squeezes everything that happened into
// the first few pixels.
const REST_SECONDS = 0.16;

export class Inspector {
  constructor({ chip, stats, canvas, legend }) {
    this.chip = chip;
    this.stats = stats;
    this.canvas = canvas;
    this.legend = legend;
    this.ctx = canvas.getContext("2d");
    this.reset();
  }

  reset() {
    this.trace = [];
    this.launchSpeed = 0;
    this.transitionAt = null;
    this.contactAt = null;
    this.contactKind = null;
    this.closed = false;
    this.restingSince = null;
    this.lastSampleAt = -Infinity;
    this.peak = 1;
  }

  beginShot(launchSpeed) {
    this.reset();
    this.launchSpeed = launchSpeed;
    this.peak = Math.max(1, launchSpeed);
  }

  /**
   * Record the cue ball. Safe to call every physics substep: the trace is
   * thinned by table time so its resolution is a property of the simulation
   * rather than of the frame rate.
   */
  sample(cue, tableTime, collisions = 0, cushions = 0) {
    if (this.closed || !cue || cue.pocketed) return;

    const v = speed(cue);
    if (v > this.peak) this.peak = v;

    if (this.contactAt === null && (collisions > 0 || cushions > 0)) {
      this.contactAt = tableTime;
      this.contactKind = collisions > 0 ? "hit a ball" : "hit a rail";
    }

    const state = motionState(cue);
    if (state === "stationary") {
      if (this.restingSince === null) this.restingSince = tableTime;
      if (tableTime - this.restingSince > REST_SECONDS) this.closed = true;
    } else {
      this.restingSince = null;
    }

    if (tableTime - this.lastSampleAt < SAMPLE_DT) return;
    this.lastSampleAt = tableTime;

    const [ux, uy] = slipVelocity(cue);
    if (this.transitionAt === null && state === "rolling" && this.trace.length > 1) {
      this.transitionAt = { t: tableTime, v };
    }
    this.trace.push({ t: tableTime, v, u: Math.hypot(ux, uy) });
  }

  /**
   * Whether 5/7·v₀ is a claim this shot can be held to.
   *
   * The prediction is for a ball decelerating freely on cloth. Once it has hit
   * a ball or a rail the speed it settles at is set by that impact, and
   * drawing the line anyway would look like the simulation missing its own
   * target when it is in fact answering a different question.
   */
  predictionApplies() {
    if (!this.launchSpeed || !this.transitionAt) return false;
    return this.contactAt === null || this.transitionAt.t < this.contactAt;
  }

  /**
   * Time axis for the plot.
   *
   * Everything worth seeing happens in the first fraction of a second, and the
   * cue ball can then roll for another five. On one linear scale the handover
   * is four pixels wide. When the tail is that lopsided the head gets its own
   * scale, the break is drawn, and both spans are named in the legend — a
   * squashed plot and an unlabelled one are both ways of not showing the data.
   */
  timeAxis() {
    const tEnd = Math.max(0.2, this.trace[this.trace.length - 1].t);
    const events = [];
    if (this.transitionAt) events.push(this.transitionAt.t);
    if (this.contactAt !== null) events.push(this.contactAt);
    if (!events.length) return { tEnd, head: null, headFrac: 1 };

    const head = Math.max(0.15, Math.max(...events) * 1.35);
    if (head > tEnd * 0.42) return { tEnd, head: null, headFrac: 1 };
    return { tEnd, head, headFrac: 0.6 };
  }

  /** Numbers for the current instant, whether or not a shot is in flight. */
  update(cue, extra = {}) {
    const live = cue && !cue.pocketed;
    const v = live ? speed(cue) : 0;
    const [ux, uy] = live ? slipVelocity(cue) : [0, 0];
    const u = Math.hypot(ux, uy);
    const state = live ? motionState(cue) : "stationary";

    this.chip.textContent = state.toUpperCase();
    this.chip.className = `state-chip ${state}`;

    const rows = [
      ["|v|", `${v.toFixed(3)} m/s`],
      ["|u| slip", `${u.toFixed(3)} m/s`],
      ["ω_z", `${(live ? cue.wz : 0).toFixed(1)} rad/s`],
      ["ω_roll", `${(live ? Math.hypot(cue.wx, cue.wy) : 0).toFixed(1)} rad/s`],
      ["table time", `${(extra.tableTime ?? 0).toFixed(2)} s`],
      ["contacts", `${extra.collisions ?? 0}b / ${extra.cushions ?? 0}r`],
    ];
    this.stats.innerHTML = rows
      .map(([k, val]) => `<div class="stat"><span class="k">${k}</span><span class="v">${val}</span></div>`)
      .join("");

    if (!this.legend) return;
    const parts = [
      `<span><i style="background:#3fb950"></i>|v| centre of mass</span>`,
      `<span><i style="background:#ffd479"></i>|u| contact-point slip</span>`,
    ];
    if (this.predictionApplies()) {
      const predicted = this.launchSpeed * ROLL_FRACTION;
      parts.push(
        `<span><i style="background:#4aa8ff"></i>5/7·v₀ = ${predicted.toFixed(2)} m/s, predicted</span>`
      );
    } else if (this.contactAt !== null) {
      parts.push(
        `<span style="color:var(--ink-faint)">${this.contactKind} before it finished sliding, so 5/7·v₀ does not apply</span>`
      );
    }
    if (this.trace.length > 1) {
      const axis = this.timeAxis();
      if (axis.head !== null) {
        parts.push(
          `<span style="color:var(--ink-faint)">axis: 0–${axis.head.toFixed(2)} s expanded, ` +
            `then ${axis.head.toFixed(2)}–${axis.tEnd.toFixed(1)} s</span>`
        );
      }
    }
    this.legend.innerHTML = parts.join("");
  }

  render() {
    const ctx = this.ctx;
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const cssW = this.canvas.parentElement.clientWidth;
    const cssH = 122;
    if (this.canvas.width !== Math.round(cssW * dpr)) {
      this.canvas.style.height = `${cssH}px`;
      this.canvas.width = Math.round(cssW * dpr);
      this.canvas.height = Math.round(cssH * dpr);
    }
    const w = this.canvas.width;
    const h = this.canvas.height;
    const pad = 6 * dpr;
    const axisH = 13 * dpr; // room under the plot for the time ticks
    const top = pad;
    const bottom = h - pad - axisH;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0d1319";
    ctx.fillRect(0, 0, w, h);

    if (this.trace.length < 2) {
      ctx.fillStyle = "#5d6b7a";
      ctx.font = `${11 * dpr}px ui-monospace, Menlo, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("take a shot to trace the cue ball", w / 2, h / 2);
      return;
    }

    const axis = this.timeAxis();
    const x0 = pad;
    const span = w - 2 * pad;
    const xSplit = x0 + span * axis.headFrac;
    const px =
      axis.head === null
        ? (t) => x0 + (t / axis.tEnd) * span
        : (t) =>
            t <= axis.head
              ? x0 + (t / axis.head) * span * axis.headFrac
              : xSplit + ((t - axis.head) / (axis.tEnd - axis.head)) * span * (1 - axis.headFrac);

    const vMax = Math.max(this.peak, 0.5) * 1.08;
    const py = (v) => bottom - (v / vMax) * (bottom - top);

    // The expanded head, tinted so the change of scale is visible before the
    // legend explains it.
    if (axis.head !== null) {
      ctx.fillStyle = "rgba(255,255,255,0.032)";
      ctx.fillRect(x0, top, xSplit - x0, bottom - top);
    }

    // The predicted rolling speed, which the measured curve should settle onto.
    if (this.predictionApplies()) {
      const y = py(this.launchSpeed * ROLL_FRACTION);
      ctx.strokeStyle = "rgba(74,168,255,0.65)";
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(w - pad, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const line = (key, colour, width) => {
      ctx.strokeStyle = colour;
      ctx.lineWidth = width * dpr;
      ctx.lineJoin = "round";
      ctx.beginPath();
      this.trace.forEach((s, i) => {
        const x = px(s.t);
        const y = py(s[key]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    line("u", "#ffd479", 1.3);
    line("v", "#3fb950", 1.7);

    const marker = (t, label, colour, labelTop) => {
      const x = px(t);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      // Stops short of the top so it does not rule through the speed scale,
      // which the handover marker otherwise does on almost every shot.
      ctx.moveTo(x, top + 10 * dpr);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.fillStyle = colour;
      ctx.font = `${9.5 * dpr}px ui-monospace, Menlo, monospace`;
      const right = x > w * 0.55;
      ctx.textAlign = right ? "right" : "left";
      ctx.textBaseline = "top";
      ctx.fillText(label, x + (right ? -4 * dpr : 4 * dpr), labelTop);
    };

    // The first text line belongs to the speed scale, so the event labels
    // start below it: the handover happens early in the shot and its label
    // would otherwise be written straight through the axis maximum.
    if (this.transitionAt) {
      marker(
        this.transitionAt.t,
        `rolls at ${this.transitionAt.v.toFixed(2)} m/s`,
        "rgba(230,237,243,0.8)",
        top + 12 * dpr
      );
    }
    // Naming the collision explains the cliff in the green curve, which
    // otherwise reads as the model losing energy for no reason.
    if (this.contactAt !== null) {
      marker(this.contactAt, this.contactKind, "rgba(240,136,62,0.9)", top + 24 * dpr);
    }

    this.drawTimeAxis(ctx, { axis, px, x0, xSplit, w, pad, bottom, top, dpr, vMax });
  }

  drawTimeAxis(ctx, { axis, px, x0, xSplit, w, pad, bottom, top, dpr, vMax }) {
    ctx.strokeStyle = "rgba(255,255,255,0.14)";
    ctx.lineWidth = 1 * dpr;
    ctx.beginPath();
    ctx.moveTo(x0, bottom);
    ctx.lineTo(w - pad, bottom);
    ctx.stroke();

    if (axis.head !== null) {
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.setLineDash([2 * dpr, 3 * dpr]);
      ctx.beginPath();
      ctx.moveTo(xSplit, top);
      ctx.lineTo(xSplit, bottom + 3 * dpr);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = "#5d6b7a";
    ctx.font = `${9 * dpr}px ui-monospace, Menlo, monospace`;
    ctx.textBaseline = "top";

    const tick = (t, align) => {
      ctx.textAlign = align;
      ctx.fillText(`${t < 10 ? t.toFixed(2) : t.toFixed(1)} s`, px(t), bottom + 3 * dpr);
    };

    ctx.textAlign = "left";
    ctx.fillText("0", x0, bottom + 3 * dpr);
    if (axis.head !== null) tick(axis.head, "center");
    tick(axis.tEnd, "right");

    ctx.textAlign = "left";
    ctx.fillText(`${vMax.toFixed(1)} m/s`, x0 + 2 * dpr, top + 1);
  }
}

export const BALL_RADIUS_MM = BALL.radius * 1000;
