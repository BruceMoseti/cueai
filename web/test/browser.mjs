/**
 * End-to-end smoke test: load the page in a real browser and play a game.
 *
 * The parity and self-play tests cover the physics and the rules without a
 * DOM, which leaves exactly one thing unchecked and it is the thing that
 * breaks: whether the modules load, wire together, and drive a canvas without
 * throwing. This plays a full game through the page's own handlers and fails
 * on any console error, unhandled rejection, or game that stops progressing.
 *
 *   node web/test/browser.mjs [--url http://localhost:8123/index.html] [--shots 40]
 *
 * Needs puppeteer-core and a Chrome binary; skips cleanly if either is absent
 * so that it never blocks a machine that only wants the dependency-free tests.
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

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, fallback) => {
    const i = args.indexOf(flag);
    return i >= 0 ? args[i + 1] : fallback;
  };
  return {
    url: get("--url", "http://localhost:8123/index.html"),
    shots: Number(get("--shots", 40)),
    games: Number(get("--games", 1)),
    screenshot: get("--screenshot", null),
    verbose: args.includes("--verbose"),
  };
}

function skip(reason) {
  console.log(`browser test skipped: ${reason}`);
  process.exit(0);
}

async function main() {
  const { url, shots: shotLimit, games, screenshot, verbose } = parseArgs();

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
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--window-size=1600,1000"],
  });

  const problems = [];
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 1 });
    page.on("console", (msg) => {
      if (msg.type() === "error" || msg.type() === "warning") {
        problems.push(`console.${msg.type()}: ${msg.text()}`);
      }
    });
    page.on("pageerror", (error) => problems.push(`uncaught: ${error.message}`));
    page.on("requestfailed", (req) =>
      problems.push(`request failed: ${req.url()} (${req.failure()?.errorText})`)
    );

    const response = await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    if (!response || !response.ok()) {
      throw new Error(`page returned ${response ? response.status() : "no response"}`);
    }

    await page.waitForFunction(() => window.pocket !== undefined, { timeout: 10000 });

    // The canvas must actually have been painted, not merely created.
    const painted = await page.evaluate(() => {
      const canvas = document.getElementById("table");
      const ctx = canvas.getContext("2d");
      const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const seen = new Set();
      for (let i = 0; i < data.length; i += 4 * 997) {
        seen.add(`${data[i]},${data[i + 1]},${data[i + 2]}`);
      }
      return { width: canvas.width, height: canvas.height, distinctColours: seen.size };
    });
    if (painted.width < 100 || painted.distinctColours < 8) {
      throw new Error(
        `canvas looks unpainted: ${painted.width}x${painted.height}, ` +
          `${painted.distinctColours} distinct colours`
      );
    }

    console.log(`page loaded and painted (${painted.width}x${painted.height} device pixels)`);
    for (let game = 0; game < games; game++) {
      if (game > 0) await page.evaluate(() => window.pocket.newGame());
      const result = await playGame(page, shotLimit, verbose);
      console.log(
        `game ${game + 1}: ${result.shots} shots (${result.youShots} yours, ` +
          `${result.botShots} the bot's), ${result.potted} balls potted, ` +
          `${result.fouls} fouls, ` +
          (result.finished ? `winner ${result.winner}` : "shot limit reached")
      );
      if (!result.finished) throw new Error("game did not reach a conclusion");
      if (result.botShots === 0) throw new Error("the bot never took a shot");
    }
    if (screenshot) await page.screenshot({ path: screenshot, fullPage: true });

    if (problems.length) {
      console.error(`\n${problems.length} console problem(s):`);
      for (const p of problems.slice(0, 20)) console.error(`  ${p}`);
      process.exit(1);
    }
    console.log("no console errors, warnings, or failed requests");
  } finally {
    await browser.close();
  }
}

/**
 * Drive the page the way a player would: place, aim at a legal ball, shoot,
 * wait for the table to settle, and let the bot have its turn.
 */
async function playGame(page, shotLimit, verbose = false) {
  let turns = 0;

  for (; turns < shotLimit; turns++) {
    const mode = await page.evaluate(() => window.pocket.mode);
    if (mode === "over") break;

    if (mode === "placing") {
      await page.evaluate(() => {
        const s = window.pocket.state;
        window.pocket.place(s.table.length * (s.behindHeadString ? 0.18 : 0.5), s.table.width * 0.5);
      });
      continue;
    }

    if (mode === "aim") {
      // Aim at a legal target through the page's own geometry, then fire. The
      // eight is only chosen once the group is genuinely cleared, so a random
      // walk does not end every game by potting it early.
      await page.evaluate(() => {
        const s = window.pocket.state;
        const cue = s.balls.find((b) => b.number === 0);
        const suit = (n) => (n === 8 ? "eight" : n <= 7 ? "solid" : "stripe");
        const mine = s.groups.you;
        const onTable = s.balls.filter((b) => !b.pocketed && b.number !== 0);
        let targets = mine
          ? onTable.filter((b) => suit(b.number) === mine)
          : onTable.filter((b) => b.number !== 8);
        if (targets.length === 0) targets = onTable.filter((b) => b.number === 8);
        if (targets.length === 0) targets = onTable;
        const t = targets[Math.floor(Math.random() * targets.length)];
        window.pocket.aim(Math.atan2(t.y - cue.y, t.x - cue.x));
        window.pocket.setPower(0.3 + Math.random() * 0.35);
        window.pocket.shoot();
      });
      await settle(page);
      continue;
    }

    if (mode === "rolling" || mode === "thinking" || mode === "stroking") {
      await settle(page);
      continue;
    }

    throw new Error(`unexpected mode "${mode}"`);
  }

  // The page records every resolved shot, which is the only reliable way to
  // see the bot's turns: they begin and end inside a single wait.
  const final = await page.evaluate(() => ({
    phase: window.pocket.state.phase,
    winner: window.pocket.state.winner,
    history: window.pocket.history,
  }));

  const history = final.history;
  if (verbose) {
    for (const [i, h] of history.entries()) {
      const label = h.wasBreak ? "break" : h.shooter;
      console.log(
        `   ${String(i + 1).padStart(2)} ${label.padEnd(5)} potted [${h.potted}]` +
          (h.foul ? ` FOUL: ${h.fouls[0]}` : "") +
          (h.continues ? " (keeps the table)" : "")
      );
    }
  }

  const byBot = history.filter((h) => h.shooter === "bot").length;
  return {
    shots: history.length,
    botShots: byBot,
    youShots: history.length - byBot,
    potted: history.reduce((a, h) => a + h.potted.filter((n) => n !== 0).length, 0),
    fouls: history.filter((h) => h.foul).length,
    finished: final.phase === "over",
    winner: final.winner,
  };
}

/** Wait until the table is at rest and the page is ready for input again. */
async function settle(page) {
  try {
    await page.waitForFunction(() => ["aim", "placing", "over"].includes(window.pocket.mode), {
      timeout: 60000,
      polling: 120,
    });
  } catch {
    // A stuck turn is the failure this test exists to catch, so say what stuck.
    const snapshot = await page.evaluate(() => {
      const s = window.pocket.state;
      const moving = s.balls
        .filter((b) => !b.pocketed && Math.hypot(b.vx, b.vy) > 1e-4)
        .map((b) => `${b.number}@${Math.hypot(b.vx, b.vy).toFixed(4)}m/s`);
      return {
        mode: window.pocket.mode,
        turn: s.turn,
        phase: s.phase,
        ballInHand: s.ballInHand,
        groups: s.groups,
        moving,
        onTable: s.balls.filter((b) => !b.pocketed).map((b) => b.number),
        bot: document.getElementById("bot-report").textContent.replace(/\s+/g, " ").trim(),
      };
    });
    throw new Error(
      `the table never settled. mode=${snapshot.mode} turn=${snapshot.turn} ` +
        `phase=${snapshot.phase} ballInHand=${snapshot.ballInHand} ` +
        `groups=${JSON.stringify(snapshot.groups)} onTable=[${snapshot.onTable}] ` +
        `moving=[${snapshot.moving}] bot="${snapshot.bot}"`
    );
  }
}

main().catch((error) => {
  console.error(`browser test failed: ${error.message}`);
  process.exit(1);
});
