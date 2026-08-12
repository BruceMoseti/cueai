/** Ball identities, colours, and the opening rack. Mirrors `cueai/physics/rack.py`. */

import { BALL, makeBall } from "./physics.js";

export const BALL_COLORS = {
  0: "#f5f5f0",
  1: "#ebc828",
  2: "#285ac8",
  3: "#c82828",
  4: "#7832a0",
  5: "#e6821e",
  6: "#1e8c3c",
  7: "#782832",
  8: "#141414",
  9: "#ebc828",
  10: "#285ac8",
  11: "#c82828",
  12: "#7832a0",
  13: "#e6821e",
  14: "#1e8c3c",
  15: "#782832",
};

export function suitOf(number) {
  if (number === 0) return "cue";
  if (number === 8) return "eight";
  return number <= 7 ? "solid" : "stripe";
}

export function footSpot(table) {
  return [table.length * 0.75, table.width * 0.5];
}

export function headSpot(table) {
  return [table.length * 0.25, table.width * 0.5];
}

/** Small deterministic PRNG so a seed reproduces a rack exactly. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(items, rand) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

/** Legal 8-ball rack: eight in the centre, a solid and a stripe in the back corners. */
export function rackOrder(seed = 7) {
  const rand = mulberry32(seed);
  const solids = shuffle([1, 2, 3, 4, 5, 6, 7], rand);
  const stripes = shuffle([9, 10, 11, 12, 13, 14, 15], rand);
  const order = new Array(15).fill(0);
  order[0] = solids.pop();
  order[4] = 8;
  order[10] = solids.pop();
  order[14] = stripes.pop();
  const rest = shuffle([...solids, ...stripes], rand);
  for (let i = 0; i < 15; i++) if (order[i] === 0) order[i] = rest.pop();
  return order;
}

export function trianglePositions(apex) {
  // Racked balls touch; see `triangle_positions` in `cueai/physics/rack.py`.
  const gap = 2 * BALL.radius;
  const spots = [];
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col <= row; col++) {
      spots.push([apex[0] + (row * gap * Math.sqrt(3)) / 2, apex[1] + (col - row / 2) * gap]);
    }
  }
  return spots;
}

export function makeRack(table, seed = 7) {
  const spots = trianglePositions(footSpot(table));
  const numbers = rackOrder(seed);
  const [cx, cy] = headSpot(table);
  const balls = [makeBall(0, cx, cy)];
  numbers.forEach((n, i) => balls.push(makeBall(n, spots[i][0], spots[i][1])));
  return balls;
}

export function cloneBalls(balls) {
  return balls.map((b) => ({ ...b }));
}
