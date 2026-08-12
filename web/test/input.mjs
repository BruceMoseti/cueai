/**
 * Drive the game with a real cursor and real keys.
 *
 * `browser.mjs` plays through `window.pocket`, which proves the modules wire
 * together but skips the layer a person actually touches: pointer capture,
 * drag thresholds, which element owns the spacebar. Those are where an
 * interactive page goes wrong, and they cannot be tested by calling the
 * functions underneath them. So this moves the mouse.
 *
 * Every assertion here is a promise the interface makes to a player:
 * a quick click lines a shot up rather than playing it, pulling the cue back
 * sets the power, the arrow keys move the aim by a hair, and the controls own
 * their own keystrokes.
 *
 *   node web/test/input.mjs [--url http://localhost:8123/index.html]
 *
 * Needs puppeteer-core and a Chrome binary; skips cleanly without either.
 */

import { existsSync } from "node:fs";

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "/usr/bin/google-chrome-stable",
  "/usr/local/bin/google-chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
].filter(Boolean);

// Mirrors of the constants in main.js. Duplicated on purpose: a test that
// imports the threshold it is checking cannot notice the threshold changing.
const CLICK_HOLD_MS = 220;
const DRAG_DEADZONE = 0.012; // metres

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, fallback) => {
    const i = args.indexOf(flag);
    return i >= 0 ? args[i + 1] : fallback;
  };
  return { url: get("--url", "http://localhost:8123/index.html") };
}

function skip(reason) {
  console.log(`input test skipped: ${reason}`);
  process.exit(0);
}

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok, detail });
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
}

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function settle(page, timeout = 60000) {
  await page.waitForFunction(() => ["aim", "placing", "over"].includes(window.pocket.mode), {
    timeout,
    polling: 100,
  });
}

/** Put the game in a known state: cue ball placed, our turn, ready to aim. */
async function resetToAim(page) {
  await page.evaluate(() => window.pocket.newGame());
  const spot = await page.evaluate(() => {
    const s = window.pocket.state;
    return window.pocket.toClient(s.table.length * 0.18, s.table.width * 0.5);
  });
  await page.mouse.click(spot.x, spot.y);
  const mode = await page.evaluate(() => window.pocket.mode);
  if (mode !== "aim") throw new Error(`placing a cue ball by clicking left mode "${mode}"`);
}

/** Viewport pixels for a point in table metres. */
function at(page, x, y) {
  return page.evaluate(([tx, ty]) => window.pocket.toClient(tx, ty), [x, y]);
}

async function cueBall(page) {
  return page.evaluate(() => {
    const b = window.pocket.state.balls.find((ball) => ball.number === 0);
    return { x: b.x, y: b.y };
  });
}

async function main() {
  const { url } = parseArgs();

  let puppeteer;
  try {
    puppeteer = (await import("puppeteer-core")).default;
  } catch {
    skip("puppeteer-core is not installed (npm i puppeteer-core)");
  }
  const executablePath = CHROME_CANDIDATES.find((p) => existsSync(p));
  if (!executablePath) skip("no Chrome binary found; set CHROME_PATH");

  const browser = await puppeteer.launch({
    executablePath,
    headless: "new",
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });

  const problems = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1500, height: 980, deviceScaleFactor: 1 });
    page.on("console", (msg) => {
      if (msg.type() === "error" || msg.type() === "warning") {
        problems.push(`console.${msg.type()}: ${msg.text()}`);
      }
    });
    page.on("pageerror", (error) => problems.push(`uncaught: ${error.message}`));

    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    await page.waitForFunction(() => window.pocket !== undefined, { timeout: 10000 });
    await page.select("#playback", "3"); // the shots here are means, not ends

    await placingPutsTheBallWhereYouClick(page);
    await theCueFollowsTheCursor(page);
    await shiftAimsSlowly(page);
    await theArrowKeysMoveByAHair(page);
    await aQuickClickDoesNotShoot(page);
    await holdingStillDoesShoot(page);
    await pullingTheCueBackSetsThePower(page);
    await theControlsOwnTheirOwnKeys(page);
    await theSpinWidgetMovesTheTip(page);
    await theShootButtonWorks(page);

    if (problems.length) {
      console.error(`\n${problems.length} console problem(s):`);
      for (const p of problems.slice(0, 20)) console.error(`  ${p}`);
      process.exit(1);
    }
  } finally {
    await browser.close();
  }

  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} interaction checks passed`);
  if (failed.length) process.exit(1);
}

// ---------- the checks ----------

async function placingPutsTheBallWhereYouClick(page) {
  await page.evaluate(() => window.pocket.newGame());
  const before = await page.evaluate(() => window.pocket.mode);
  const target = await page.evaluate(() => {
    const s = window.pocket.state;
    return { x: s.table.length * 0.18, y: s.table.width * 0.36 };
  });
  const spot = await at(page, target.x, target.y);
  await page.mouse.click(spot.x, spot.y);
  const cue = await cueBall(page);
  const mode = await page.evaluate(() => window.pocket.mode);
  const off = Math.hypot(cue.x - target.x, cue.y - target.y);
  check(
    "clicking behind the head string places the cue ball there",
    before === "placing" && mode === "aim" && off < 0.01,
    `${(off * 1000).toFixed(1)} mm from the click, mode "${mode}"`
  );

  // In front of the head string is illegal on the break, and the page snaps
  // back to the nearest legal spot rather than silently ignoring the click.
  await page.evaluate(() => window.pocket.newGame());
  const illegal = await page.evaluate(() => {
    const s = window.pocket.state;
    return { x: s.table.length * 0.8, y: s.table.width * 0.5 };
  });
  const far = await at(page, illegal.x, illegal.y);
  await page.mouse.click(far.x, far.y);
  const after = await cueBall(page);
  const headString = await page.evaluate(() => window.pocket.state.table.length * 0.25);
  check(
    "an illegal placement snaps behind the head string",
    after.x <= headString + 1e-6,
    `landed at x=${after.x.toFixed(3)} m, head string at ${headString.toFixed(3)} m`
  );
}

async function theCueFollowsTheCursor(page) {
  await resetToAim(page);
  const cue = await cueBall(page);
  const table = await page.evaluate(() => window.pocket.state.table);

  const targets = [
    [cue.x + 0.5, cue.y],
    [cue.x + 0.4, cue.y + 0.4],
    [cue.x + 0.3, cue.y - 0.45],
  ];
  let worst = 0;
  for (const [tx, ty] of targets) {
    const p = await at(page, Math.min(tx, table.length - 0.05), Math.max(0.05, ty));
    await page.mouse.move(p.x, p.y);
    const angle = await page.evaluate(() => window.pocket.aimAngle);
    const want = Math.atan2(Math.max(0.05, ty) - cue.y, Math.min(tx, table.length - 0.05) - cue.x);
    let delta = Math.abs(angle - want);
    if (delta > Math.PI) delta = 2 * Math.PI - delta;
    worst = Math.max(worst, delta);
  }
  check(
    "moving the mouse aims at the cursor",
    worst < 0.01,
    `worst ${((worst * 180) / Math.PI).toFixed(2)}° from the cursor`
  );
}

async function shiftAimsSlowly(page) {
  await resetToAim(page);
  const cue = await cueBall(page);
  const start = await at(page, cue.x + 0.5, cue.y);
  await page.mouse.move(start.x, start.y);
  const before = await page.evaluate(() => window.pocket.aimAngle);

  // One shift-move a long way round: the aim should ease toward the cursor
  // rather than snap to it, which is what makes a quarter-degree cut reachable.
  const away = await at(page, cue.x + 0.35, cue.y + 0.35);
  await page.keyboard.down("Shift");
  await page.mouse.move(away.x, away.y);
  await page.keyboard.up("Shift");
  const after = await page.evaluate(() => window.pocket.aimAngle);

  const demanded = Math.atan2(cue.y + 0.35 - cue.y, cue.x + 0.35 - cue.x) - before;
  const moved = after - before;
  const fraction = moved / demanded;
  check(
    "shift eases the aim instead of snapping it",
    fraction > 0 && fraction < 0.35,
    `moved ${(fraction * 100).toFixed(0)}% of the way to the cursor`
  );
}

async function theArrowKeysMoveByAHair(page) {
  await resetToAim(page);
  await page.evaluate(() => document.body.focus());

  const a0 = await page.evaluate(() => window.pocket.aimAngle);
  await page.keyboard.press("ArrowRight");
  const a1 = await page.evaluate(() => window.pocket.aimAngle);
  await page.keyboard.down("Shift");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.up("Shift");
  const a2 = await page.evaluate(() => window.pocket.aimAngle);

  const coarse = a1 - a0;
  const fine = a2 - a1;
  check(
    "the arrow keys nudge the aim, and shift nudges it less",
    coarse > 0 && fine > 0 && fine < coarse / 3,
    `${((coarse * 180) / Math.PI).toFixed(3)}° plain, ${((fine * 180) / Math.PI).toFixed(3)}° with shift`
  );

  const p0 = await page.evaluate(() => window.pocket.power);
  await page.keyboard.press("ArrowUp");
  const p1 = await page.evaluate(() => window.pocket.power);
  const readout = await page.evaluate(() => document.getElementById("power-readout").textContent);
  check(
    "the up arrow raises the power, and the readout follows",
    p1 > p0 && readout.includes(((p1 * 7.5).toFixed(1))),
    `${p0.toFixed(2)} to ${p1.toFixed(2)}, readout "${readout}"`
  );
}

async function aQuickClickDoesNotShoot(page) {
  await resetToAim(page);
  const cue = await cueBall(page);
  const p = await at(page, cue.x + 0.35, cue.y + 0.02);

  await page.mouse.move(p.x, p.y);
  await page.mouse.down();
  await page.mouse.up();
  await wait(120);

  const mode = await page.evaluate(() => window.pocket.mode);
  const shots = await page.evaluate(() => window.pocket.history.length);
  check(
    "a quick click lines the shot up rather than playing it",
    mode === "aim" && shots === 0,
    `mode "${mode}" after the click, ${shots} shots played`
  );
}

async function holdingStillDoesShoot(page) {
  await resetToAim(page);
  const cue = await cueBall(page);
  const p = await at(page, cue.x + 0.35, cue.y + 0.02);

  await page.mouse.move(p.x, p.y);
  await page.mouse.down();
  await wait(CLICK_HOLD_MS + 140);
  await page.mouse.up();
  await wait(120);

  const mode = await page.evaluate(() => window.pocket.mode);
  check(
    "a press held on the spot plays the shot",
    ["stroking", "rolling"].includes(mode),
    `mode "${mode}" on release`
  );
  await settle(page);
}

async function pullingTheCueBackSetsThePower(page) {
  await resetToAim(page);
  const cue = await cueBall(page);

  // Aim along +x, then drag the cursor back along -x: that is the cue being
  // drawn, and the further it comes back the harder the shot.
  const ahead = await at(page, cue.x + 0.5, cue.y);
  await page.mouse.move(ahead.x, ahead.y);
  const angle = await page.evaluate(() => window.pocket.aimAngle);
  const before = await page.evaluate(() => window.pocket.power);

  await page.mouse.down();
  const short = await at(page, cue.x + 0.5 - DRAG_DEADZONE * 0.4, cue.y);
  await page.mouse.move(short.x, short.y);
  const nudged = await page.evaluate(() => window.pocket.power);
  check(
    "a twitch inside the dead zone does not change the power",
    Math.abs(nudged - before) < 1e-9,
    `power ${nudged.toFixed(3)}`
  );

  const drawn = await at(page, cue.x + 0.5 - 0.25, cue.y);
  await page.mouse.move(drawn.x, drawn.y, { steps: 6 });
  const pulled = await page.evaluate(() => window.pocket.power);
  const aimHeld = await page.evaluate(() => window.pocket.aimAngle);
  check(
    "drawing the cue back raises the power without disturbing the aim",
    pulled > before + 0.1 && Math.abs(aimHeld - angle) < 1e-9,
    `power ${before.toFixed(2)} to ${pulled.toFixed(2)}, aim unchanged`
  );

  await page.mouse.up();
  await wait(120);
  const mode = await page.evaluate(() => window.pocket.mode);
  check(
    "releasing a drawn cue plays the shot",
    ["stroking", "rolling"].includes(mode),
    `mode "${mode}" on release`
  );
  await settle(page);
}

async function theControlsOwnTheirOwnKeys(page) {
  await resetToAim(page);

  // Space belongs to whichever control has focus. A dropdown that swallows the
  // spacebar to open itself must not also fire a shot.
  await page.focus("#difficulty");
  await page.keyboard.press("Space");
  await wait(120);
  const afterSelect = await page.evaluate(() => window.pocket.mode);
  check(
    "space in a dropdown does not play a shot",
    afterSelect === "aim",
    `mode "${afterSelect}"`
  );

  await page.evaluate(() => document.activeElement.blur());
  await page.keyboard.press("Space");
  await wait(120);
  const afterTable = await page.evaluate(() => window.pocket.mode);
  check(
    "space with nothing focused plays the shot",
    ["stroking", "rolling"].includes(afterTable),
    `mode "${afterTable}"`
  );
  await settle(page);
}

async function theSpinWidgetMovesTheTip(page) {
  await resetToAim(page);
  const box = await (await page.$("#spin")).boundingBox();
  const before = await page.evaluate(() => window.pocket.spin);

  // Bottom of the circle: draw. Then drag well outside it, which must clamp to
  // the miscue limit rather than let the tip leave the ball.
  await page.mouse.move(box.x + box.width / 2, box.y + box.height * 0.78);
  await page.mouse.down();
  const drawn = await page.evaluate(() => window.pocket.spin);
  await page.mouse.move(box.x + box.width * 2, box.y + box.height / 2, { steps: 4 });
  const clamped = await page.evaluate(() => window.pocket.spin);
  await page.mouse.up();

  const magnitude = Math.hypot(clamped.x, clamped.y);
  check(
    "the spin widget moves the tip and clamps at the miscue limit",
    before.x === 0 && before.y === 0 && drawn.y < -0.1 && magnitude <= 0.5 + 1e-9,
    `draw y=${drawn.y.toFixed(2)}, clamped magnitude ${magnitude.toFixed(3)}`
  );
}

async function theShootButtonWorks(page) {
  await resetToAim(page);
  const disabledBefore = await page.evaluate(() => document.getElementById("shoot").disabled);
  await page.click("#shoot");
  await wait(120);
  const mode = await page.evaluate(() => window.pocket.mode);
  const disabledDuring = await page.evaluate(() => document.getElementById("shoot").disabled);
  check(
    "the shoot button plays a shot and locks while the balls roll",
    !disabledBefore && ["stroking", "rolling"].includes(mode) && disabledDuring,
    `mode "${mode}"`
  );
  await settle(page);
}

main().catch((error) => {
  console.error(`input test failed: ${error.message}`);
  process.exit(1);
});
