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

export class Inspector {
  constructor({ chip, stats, canvas, legend }) {
    this.chip = chip;
    this.stats = stats;
    this.canvas = canvas;
    this.legend = legend;
    this.ctx = canvas.getContext("2d");
    this.trace = [];
    this.launchSpeed = 0;
    this.transitionAt = null;
    this.reset();
  }

  reset() {
    this.trace = [];
    this.launchSpeed = 0;
    this.transitionAt = null;
    this.peak = 1;
  }

  beginShot(launchSpeed) {
    this.reset();
    this.launchSpeed = launchSpeed;
    this.peak = Math.max(1, launchSpeed);
  }

  sample(cue, tableTime) {
    if (!cue || cue.pocketed) return;
    const [ux, uy] = slipVelocity(cue);
    const v = speed(cue);
    const u = Math.hypot(ux, uy);
    const state = motionState(cue);
    if (this.transitionAt === null && state === "rolling" && this.trace.length > 1) {
      this.transitionAt = { t: tableTime, v };
    }
    this.trace.push({ t: tableTime, v, u, wz: cue.wz, state });
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
      ["contacts", `${extra.collisions ?? 0} ball, ${extra.cushions ?? 0} rail`],
    ];
    this.stats.innerHTML = rows
      .map(([k, val]) => `<div class="stat"><span class="k">${k}</span><span class="v">${val}</span></div>`)
      .join("");

    if (this.legend) {
      const predicted = this.launchSpeed * ROLL_FRACTION;
      this.legend.innerHTML =
        `<span><i style="background:#3fb950"></i>|v| centre of mass</span>` +
        `<span><i style="background:#ffd479"></i>|u| contact-point slip</span>` +
        (this.launchSpeed > 0
          ? `<span><i style="background:#4aa8ff"></i>5/7·v₀ = ${predicted.toFixed(2)} m/s</span>`
          : "");
    }
  }

  render() {
    const ctx = this.ctx;
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const cssW = this.canvas.parentElement.clientWidth;
    const cssH = 104;
    if (this.canvas.width !== Math.round(cssW * dpr)) {
      this.canvas.style.height = `${cssH}px`;
      this.canvas.width = Math.round(cssW * dpr);
      this.canvas.height = Math.round(cssH * dpr);
    }
    const w = this.canvas.width;
    const h = this.canvas.height;
    const pad = 6 * dpr;

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

    const tMax = Math.max(0.5, this.trace[this.trace.length - 1].t);
    const vMax = Math.max(this.peak, 0.5) * 1.08;
    const px = (t) => pad + (t / tMax) * (w - 2 * pad);
    const py = (v) => h - pad - (v / vMax) * (h - 2 * pad);

    // The predicted rolling speed, which the measured curve should settle onto.
    if (this.launchSpeed > 0) {
      const y = py(this.launchSpeed * ROLL_FRACTION);
      ctx.strokeStyle = "rgba(74,168,255,0.65)";
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(pad, y);
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

    if (this.transitionAt) {
      const x = px(this.transitionAt.t);
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 1 * dpr;
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, h - pad);
      ctx.stroke();
      ctx.fillStyle = "rgba(230,237,243,0.8)";
      ctx.font = `${9.5 * dpr}px ui-monospace, Menlo, monospace`;
      ctx.textAlign = x > w * 0.6 ? "right" : "left";
      ctx.textBaseline = "top";
      ctx.fillText(
        ` rolls at ${this.transitionAt.v.toFixed(2)} m/s `,
        x + (x > w * 0.6 ? -3 * dpr : 3 * dpr),
        pad + 1
      );
    }
  }
}

/** Peak speed seen so far, so the plot's vertical scale never clips. */
export function trackPeak(inspector, cue) {
  if (!cue || cue.pocketed) return;
  const v = speed(cue);
  if (v > inspector.peak) inspector.peak = v;
}

export const BALL_RADIUS_MM = BALL.radius * 1000;
