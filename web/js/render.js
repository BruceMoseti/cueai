/**
 * Canvas rendering for the table.
 *
 * Pure drawing: everything here reads game state and view state and writes
 * pixels, so the physics and the rules never have to know a screen exists.
 */

import { BALL } from "./physics.js";
import { BALL_COLORS, suitOf } from "./rack.js";

const RAIL = 0.092; // metres of woodwork drawn outside the playing surface
const DIAMOND_INSET = 0.046;

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.scale = 1;
    this.dpr = 1;
  }

  /** Table dimensions including the rails, in metres. */
  outerSize(table) {
    return [table.length + 2 * RAIL, table.width + 2 * RAIL];
  }

  resize(table) {
    const [ow, oh] = this.outerSize(table);
    const cssWidth = this.canvas.parentElement.clientWidth;
    const cssHeight = (cssWidth * oh) / ow;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    this.canvas.style.height = `${cssHeight}px`;
    this.canvas.width = Math.round(cssWidth * this.dpr);
    this.canvas.height = Math.round(cssHeight * this.dpr);
    this.scale = (cssWidth * this.dpr) / ow;
  }

  toCanvas(x, y) {
    return [(x + RAIL) * this.scale, (y + RAIL) * this.scale];
  }

  /** Screen pixels (CSS, relative to the canvas) back to table metres. */
  toTable(px, py) {
    const rect = this.canvas.getBoundingClientRect();
    const sx = (px / rect.width) * this.canvas.width;
    const sy = (py / rect.height) * this.canvas.height;
    return [sx / this.scale - RAIL, sy / this.scale - RAIL];
  }

  draw(state, view) {
    const ctx = this.ctx;
    const table = state.table;
    ctx.save();
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this.drawFrame(table);
    this.drawFelt(table, state, view);
    this.drawPockets(table);

    for (const b of state.balls) {
      if (!b.pocketed) this.drawShadow(b);
    }

    if (view.aim && view.showAim) this.drawAimOverlay(state, view);

    for (const b of state.balls) {
      if (b.pocketed) continue;
      const legal = view.highlight?.has(b.number);
      this.drawBall(b, legal, view);
    }

    if (view.ghostCue) this.drawGhostCue(view.ghostCue);
    if (view.aim && view.showAim && view.showCue) this.drawCueStick(state, view);

    ctx.restore();
  }

  // ---------- table furniture ----------

  drawFrame(table) {
    const ctx = this.ctx;
    const [ow, oh] = this.outerSize(table);
    const w = ow * this.scale;
    const h = oh * this.scale;
    const r = 0.02 * this.scale * 100;

    const wood = ctx.createLinearGradient(0, 0, 0, h);
    wood.addColorStop(0, "#4a3324");
    wood.addColorStop(0.5, "#38251a");
    wood.addColorStop(1, "#241610");
    ctx.fillStyle = wood;
    roundRect(ctx, 0, 0, w, h, r);
    ctx.fill();

    // Inner bevel where the woodwork meets the cushion.
    const inset = RAIL * 0.28 * this.scale;
    ctx.strokeStyle = "rgba(0,0,0,0.35)";
    ctx.lineWidth = Math.max(1, 0.004 * this.scale);
    roundRect(ctx, inset, inset, w - 2 * inset, h - 2 * inset, r * 0.6);
    ctx.stroke();

    this.drawDiamonds(table);
  }

  drawDiamonds(table) {
    const ctx = this.ctx;
    const size = 0.0075 * this.scale;
    ctx.fillStyle = "rgba(232, 220, 190, 0.72)";
    const marks = [];
    for (let i = 1; i <= 7; i++) {
      if (i === 4) continue; // the side pocket sits where the middle sight would
      const x = (table.length * i) / 8;
      marks.push([x, -DIAMOND_INSET], [x, table.width + DIAMOND_INSET]);
    }
    for (let i = 1; i <= 3; i++) {
      const y = (table.width * i) / 4;
      marks.push([-DIAMOND_INSET, y], [table.length + DIAMOND_INSET, y]);
    }
    for (const [mx, my] of marks) {
      const [px, py] = this.toCanvas(mx, my);
      ctx.beginPath();
      ctx.moveTo(px, py - size);
      ctx.lineTo(px + size, py);
      ctx.lineTo(px, py + size);
      ctx.lineTo(px - size, py);
      ctx.closePath();
      ctx.fill();
    }
  }

  drawFelt(table, state, view) {
    const ctx = this.ctx;
    const [x0, y0] = this.toCanvas(0, 0);
    const w = table.length * this.scale;
    const h = table.width * this.scale;

    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, w, h);
    ctx.clip();

    ctx.fillStyle = "#12594a";
    ctx.fillRect(x0, y0, w, h);

    // A soft centre light, which is what stops the felt reading as flat paint.
    const glow = ctx.createRadialGradient(
      x0 + w / 2,
      y0 + h / 2,
      h * 0.05,
      x0 + w / 2,
      y0 + h / 2,
      w * 0.62
    );
    glow.addColorStop(0, "rgba(255,255,255,0.10)");
    glow.addColorStop(0.55, "rgba(255,255,255,0.02)");
    glow.addColorStop(1, "rgba(0,0,0,0.30)");
    ctx.fillStyle = glow;
    ctx.fillRect(x0, y0, w, h);

    // Head string, drawn only while it constrains where the cue ball may go.
    if (state.behindHeadString && (state.ballInHand || state.phase === "break")) {
      const [hx] = this.toCanvas(table.length * 0.25, 0);
      ctx.strokeStyle = "rgba(255,255,255,0.22)";
      ctx.setLineDash([6 * this.dpr, 6 * this.dpr]);
      ctx.lineWidth = Math.max(1, 0.0022 * this.scale);
      ctx.beginPath();
      ctx.moveTo(hx, y0);
      ctx.lineTo(hx, y0 + h);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (view.footSpot) {
      const [fx, fy] = this.toCanvas(table.length * 0.75, table.width * 0.5);
      ctx.fillStyle = "rgba(255,255,255,0.16)";
      ctx.beginPath();
      ctx.arc(fx, fy, 0.004 * this.scale, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();

    ctx.strokeStyle = "rgba(0,0,0,0.5)";
    ctx.lineWidth = Math.max(1, 0.003 * this.scale);
    ctx.strokeRect(x0, y0, w, h);
  }

  drawPockets(table) {
    const ctx = this.ctx;
    for (const [px, py] of table.pockets) {
      const [cx, cy] = this.toCanvas(px, py);
      const r = table.pocketRadius * this.scale;
      const grad = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r);
      grad.addColorStop(0, "#000");
      grad.addColorStop(0.75, "#05070a");
      grad.addColorStop(1, "#1d2a22");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.lineWidth = Math.max(1, 0.0025 * this.scale);
      ctx.stroke();
    }
  }

  // ---------- balls ----------

  drawShadow(b) {
    const ctx = this.ctx;
    const [cx, cy] = this.toCanvas(b.x, b.y);
    const r = BALL.radius * this.scale;
    ctx.fillStyle = "rgba(0,0,0,0.34)";
    ctx.beginPath();
    ctx.ellipse(cx + r * 0.16, cy + r * 0.24, r * 1.02, r * 0.94, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  drawBall(b, legal, view) {
    const ctx = this.ctx;
    const [cx, cy] = this.toCanvas(b.x, b.y);
    const r = BALL.radius * this.scale;
    const colour = BALL_COLORS[b.number];
    const stripe = suitOf(b.number) === "stripe";
    // The in-plane component of the ball's rotation is the part a top-down
    // view can honestly show, so the markings turn with english.
    const spin = b.visualSpin ?? 0;

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.clip();

    ctx.fillStyle = stripe ? "#f4f2e9" : colour;
    ctx.fillRect(cx - r, cy - r, 2 * r, 2 * r);

    if (stripe) {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(spin);
      ctx.fillStyle = colour;
      ctx.fillRect(-r, -r * 0.56, 2 * r, r * 1.12);
      ctx.restore();
    }

    if (b.number !== 0) {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(spin);
      ctx.fillStyle = "#f7f5ee";
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.44, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#15181c";
      ctx.font = `700 ${r * (b.number > 9 ? 0.5 : 0.62)}px ${
        "ui-monospace, Menlo, Consolas, monospace"
      }`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(b.number), 0, r * 0.03);
      ctx.restore();
    }

    // Shading and a specular dot, which is most of what sells a sphere.
    const shade = ctx.createRadialGradient(
      cx - r * 0.34,
      cy - r * 0.4,
      r * 0.1,
      cx,
      cy,
      r * 1.18
    );
    shade.addColorStop(0, "rgba(255,255,255,0.42)");
    shade.addColorStop(0.42, "rgba(255,255,255,0.03)");
    shade.addColorStop(0.78, "rgba(0,0,0,0.16)");
    shade.addColorStop(1, "rgba(0,0,0,0.55)");
    ctx.fillStyle = shade;
    ctx.fillRect(cx - r, cy - r, 2 * r, 2 * r);

    ctx.beginPath();
    ctx.arc(cx - r * 0.33, cy - r * 0.38, r * 0.15, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.62)";
    ctx.fill();
    ctx.restore();

    ctx.strokeStyle = "rgba(0,0,0,0.45)";
    ctx.lineWidth = Math.max(1, 0.0015 * this.scale);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    if (legal && view.showTargets) {
      ctx.strokeStyle = "rgba(74,168,255,0.85)";
      ctx.lineWidth = Math.max(1.2, 0.0022 * this.scale);
      ctx.setLineDash([3 * this.dpr, 3 * this.dpr]);
      ctx.beginPath();
      ctx.arc(cx, cy, r * 1.32, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  drawGhostCue({ x, y, legal }) {
    const ctx = this.ctx;
    const [cx, cy] = this.toCanvas(x, y);
    const r = BALL.radius * this.scale;
    ctx.fillStyle = legal ? "rgba(245,245,240,0.5)" : "rgba(248,81,73,0.35)";
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = legal ? "rgba(255,255,255,0.85)" : "rgba(248,81,73,0.95)";
    ctx.lineWidth = Math.max(1.2, 0.002 * this.scale);
    ctx.setLineDash([4 * this.dpr, 4 * this.dpr]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // ---------- aiming ----------

  drawAimOverlay(state, view) {
    const ctx = this.ctx;
    const { aim } = view;
    const cue = state.balls.find((b) => b.number === 0);
    if (!cue || cue.pocketed) return;

    const r = BALL.radius * this.scale;
    const [sx, sy] = this.toCanvas(cue.x, cue.y);
    const [ex, ey] = this.toCanvas(aim.x, aim.y);

    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = Math.max(1, 0.0018 * this.scale);
    ctx.setLineDash([7 * this.dpr, 5 * this.dpr]);
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.setLineDash([]);

    if (!aim.hit) return;

    // Ghost ball: where the cue ball's centre sits at contact.
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.lineWidth = Math.max(1, 0.0016 * this.scale);
    ctx.setLineDash([3 * this.dpr, 3 * this.dpr]);
    ctx.beginPath();
    ctx.arc(ex, ey, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    const [ox, oy] = this.toCanvas(aim.hit.x, aim.hit.y);
    const objLen = 0.42 * this.scale;
    ctx.strokeStyle = BALL_COLORS[aim.hit.number];
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = Math.max(1.4, 0.0026 * this.scale);
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.lineTo(ox + aim.objectDir[0] * objLen, oy + aim.objectDir[1] * objLen);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // The cue ball leaves along the tangent, perpendicular to the line of
    // centres. Showing both lines makes the 90 degree rule visible.
    const tanLen = 0.2 * this.scale;
    const side = view.tangentSide ?? 1;
    ctx.strokeStyle = "rgba(180,210,255,0.5)";
    ctx.lineWidth = Math.max(1, 0.0016 * this.scale);
    ctx.setLineDash([4 * this.dpr, 4 * this.dpr]);
    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(ex + aim.tangentDir[0] * tanLen * side, ey + aim.tangentDir[1] * tanLen * side);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawCueStick(state, view) {
    const ctx = this.ctx;
    const cue = state.balls.find((b) => b.number === 0);
    if (!cue || cue.pocketed) return;

    const r = BALL.radius * this.scale;
    const [cx, cy] = this.toCanvas(cue.x, cue.y);
    const angle = view.angle;
    // Pull back with power, and add the tip offset so the stick visibly aims
    // at the part of the ball the spin controls select.
    const gap = r * (1.25 + 3.4 * view.power);
    const length = 1.45 * this.scale;
    const perp = -view.spin.x * r * 0.85;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);
    ctx.translate(0, perp);

    ctx.fillStyle = "rgba(0,0,0,0.3)";
    ctx.beginPath();
    ctx.moveTo(-gap, -r * 0.1 + r * 0.2);
    ctx.lineTo(-gap - length, -r * 0.26 + r * 0.2);
    ctx.lineTo(-gap - length, r * 0.26 + r * 0.2);
    ctx.lineTo(-gap, r * 0.1 + r * 0.2);
    ctx.closePath();
    ctx.fill();

    const shaft = ctx.createLinearGradient(-gap, -r * 0.2, -gap, r * 0.2);
    shaft.addColorStop(0, "#f0d9a8");
    shaft.addColorStop(0.42, "#d8b57a");
    shaft.addColorStop(1, "#8d6437");
    ctx.fillStyle = shaft;
    ctx.beginPath();
    ctx.moveTo(-gap, -r * 0.1);
    ctx.lineTo(-gap - length, -r * 0.26);
    ctx.lineTo(-gap - length, r * 0.26);
    ctx.lineTo(-gap, r * 0.1);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = "#2b2a30";
    ctx.beginPath();
    ctx.moveTo(-gap - length * 0.58, -r * 0.2);
    ctx.lineTo(-gap - length, -r * 0.26);
    ctx.lineTo(-gap - length, r * 0.26);
    ctx.lineTo(-gap - length * 0.58, r * 0.2);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = "#5fa8d3";
    ctx.beginPath();
    ctx.moveTo(-gap, -r * 0.1);
    ctx.lineTo(-gap - r * 0.16, -r * 0.11);
    ctx.lineTo(-gap - r * 0.16, r * 0.11);
    ctx.lineTo(-gap, r * 0.1);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

/** The spin selector: a cue ball you click to move the tip off centre. */
export function drawSpinWidget(canvas, spin) {
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const size = 74;
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  const c = (size * dpr) / 2;
  const r = c * 0.86;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const grad = ctx.createRadialGradient(c - r * 0.3, c - r * 0.35, r * 0.1, c, c, r);
  grad.addColorStop(0, "#ffffff");
  grad.addColorStop(0.6, "#e8e6dd");
  grad.addColorStop(1, "#a9a79c");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(c, c, r, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(0,0,0,0.35)";
  ctx.lineWidth = 1 * dpr;
  ctx.beginPath();
  ctx.moveTo(c - r, c);
  ctx.lineTo(c + r, c);
  ctx.moveTo(c, c - r);
  ctx.lineTo(c, c + r);
  ctx.stroke();

  // Past half a radius the tip slides off the ball: the miscue limit.
  ctx.strokeStyle = "rgba(200,60,50,0.55)";
  ctx.setLineDash([3 * dpr, 3 * dpr]);
  ctx.beginPath();
  ctx.arc(c, c, r * 0.5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  const tx = c + spin.x * r;
  const ty = c - spin.y * r;
  ctx.fillStyle = "#4aa8ff";
  ctx.strokeStyle = "#0b0f14";
  ctx.lineWidth = 1.6 * dpr;
  ctx.beginPath();
  ctx.arc(tx, ty, r * 0.17, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}
