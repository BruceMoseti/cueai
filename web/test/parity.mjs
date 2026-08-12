/**
 * Check the browser physics against the Python reference simulator.
 *
 * `scripts/export_parity_cases.py` runs a spread of shots through
 * `src/cueai/physics/` and records where every ball came to rest. This replays
 * the same shots through the module the game actually uses and reports the
 * worst disagreement. A port that is not measured is a rumour.
 *
 *   node web/test/parity.mjs [--verbose]
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { applyShot, defaultTable, makeBall, simulateToRest } from "../js/physics.js";

const HERE = dirname(fileURLToPath(import.meta.url));
// Ports are compared in millimetres because that is the scale the game and the
// published error numbers live at; a pool ball is 57 mm across.
const TOLERANCE_MM = 1.0;

function runCase(testCase, dt, maxTime) {
  const table = defaultTable();
  const balls = testCase.balls.map((b) => makeBall(b.number, b.x, b.y));
  const cue = balls.find((b) => b.number === 0);
  const shot = testCase.shot;
  applyShot(cue, {
    speed: shot.speed,
    angle: shot.angle,
    englishX: shot.english_x,
    englishY: shot.english_y,
  });
  if (Math.abs(shot.cue_elevation) > 1e-6) {
    cue.wz += (0.35 * shot.cue_elevation * shot.speed) / 0.028575;
  }
  const events = simulateToRest(balls, table, { dt, maxTime });
  return { balls, events };
}

function compare(testCase, result) {
  const ref = testCase.reference;
  const pocketed = result.balls.filter((b) => b.pocketed).map((b) => b.number).sort((a, b) => a - b);
  const refPocketed = [...ref.pocketed].sort((a, b) => a - b);

  if (JSON.stringify(pocketed) !== JSON.stringify(refPocketed)) {
    return {
      ok: false,
      worstMm: Infinity,
      detail: `pocketed [${pocketed}] but reference pocketed [${refPocketed}]`,
    };
  }

  let worstMm = 0;
  let worstBall = null;
  for (const b of result.balls) {
    if (b.pocketed) continue;
    const target = ref.resting[String(b.number)];
    if (!target) return { ok: false, worstMm: Infinity, detail: `no reference for ball ${b.number}` };
    const mm = Math.hypot(b.x - target[0], b.y - target[1]) * 1000;
    if (mm > worstMm) {
      worstMm = mm;
      worstBall = b.number;
    }
  }
  // A picometre nudge to the reference's own input moves it this far, so it is
  // the floor on any agreement two implementations can reach.
  const floorMm = testCase.chaos_yardstick_m * 1000;
  const budgetMm = Math.max(TOLERANCE_MM, floorMm);
  return {
    ok: worstMm <= budgetMm,
    worstMm,
    worstBall,
    budgetMm,
    detail: `ball ${worstBall} off by ${worstMm.toFixed(4)} mm (budget ${budgetMm.toFixed(4)} mm)`,
  };
}

function main() {
  const verbose = process.argv.includes("--verbose");
  const payload = JSON.parse(readFileSync(join(HERE, "parity_cases.json"), "utf8"));
  const { dt, max_time: maxTime, cases } = payload;

  let failures = 0;
  let worstOverall = 0;
  let worstName = "";
  let jsSeconds = 0;
  let pySeconds = 0;
  let tableSeconds = 0;
  const started = Date.now();

  for (const testCase of cases) {
    const t0 = process.hrtime.bigint();
    const result = runCase(testCase, dt, maxTime);
    jsSeconds += Number(process.hrtime.bigint() - t0) / 1e9;
    pySeconds += testCase.reference.seconds ?? 0;
    tableSeconds += testCase.reference.table_time ?? 0;
    const verdict = compare(testCase, result);
    if (Number.isFinite(verdict.worstMm) && verdict.worstMm > worstOverall) {
      worstOverall = verdict.worstMm;
      worstName = testCase.name;
    }
    if (!verdict.ok) {
      failures++;
      console.error(`FAIL ${testCase.name}: ${verdict.detail}`);
    } else if (verbose) {
      console.log(
        `ok   ${testCase.name.padEnd(28)} ${verdict.worstMm.toFixed(5)} mm  ` +
          `(${result.events.collisions} collisions, ${result.events.cushions} cushions, ` +
          `${result.events.potted.length} potted)`
      );
    }
  }

  const elapsed = ((Date.now() - started) / 1000).toFixed(1);
  console.log(
    `\n${cases.length - failures}/${cases.length} shots agree with the Python reference ` +
      `within ${TOLERANCE_MM} mm (${elapsed}s)`
  );
  console.log(`worst disagreement: ${worstOverall.toExponential(2)} mm on ${worstName}`);
  if (pySeconds > 0) {
    // Identical shots, identical timestep, identical outcomes: the only thing
    // that differs is the language, so the ratio means something.
    console.log(
      `same ${tableSeconds.toFixed(0)} s of table time: ` +
        `${pySeconds.toFixed(1)} s in Python, ${jsSeconds.toFixed(2)} s here ` +
        `(${(pySeconds / jsSeconds).toFixed(0)}x)`
    );
  }

  // The page quotes these numbers. Writing them here rather than typing them
  // into the HTML is the difference between a measurement and a claim.
  const out = join(HERE, "..", "data", "parity.json");
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(
    out,
    `${JSON.stringify(
      {
        cases: cases.length,
        agreed: cases.length - failures,
        worst_mm: worstOverall,
        worst_case: worstName,
        table_seconds: tableSeconds,
        python_seconds: pySeconds,
        browser_seconds: jsSeconds,
        speedup: pySeconds / jsSeconds,
      },
      null,
      2
    )}\n`
  );

  if (failures > 0) {
    console.error(`\n${failures} case(s) diverged. The browser port no longer matches the reference.`);
    process.exit(1);
  }
}

main();
