/**
 * Record the running game, for the README.
 *
 * A screenshot of a physics engine is worth very little and a claim about one
 * is worth less, so the images and the clip in the documentation are produced
 * by driving the real page in a real browser rather than assembled by hand.
 * Re-running this after a change to the physics or the interface updates them.
 *
 *   node web/test/capture.mjs --url http://localhost:8123/index.html
 *
 * Needs puppeteer-core, a Chrome binary and ffmpeg; skips cleanly without them.
 */

import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "/usr/bin/google-chrome-stable",
  "/usr/local/bin/google-chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
].filter(Boolean);

// Long enough to get past the break and through several shots on an open
// table. A clip that is mostly one break says very little about the game, and
// one that costs two megabytes is a README nobody scrolls.
const GIF_SECONDS = 15;
// How much of each of the bot's searches to leave in. Enough to read as a
// pause for thought; not the two seconds it actually takes at full strength.
const THINKING_SECONDS = 0.8;

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, fallback) => {
    const i = args.indexOf(flag);
    return i >= 0 ? args[i + 1] : fallback;
  };
  return {
    url: get("--url", "http://localhost:8123/index.html"),
    out: get("--out", "docs/assets"),
    seconds: Number(get("--seconds", 24)),
    noVideo: args.includes("--no-video"),
  };
}

function skip(reason) {
  console.log(`capture skipped: ${reason}`);
  process.exit(0);
}

/** Aim at the ghost ball for the straightest clear pot, the way a player would. */
const AIM_AT_BEST_POT = () => {
  const s = window.pocket.state;
  const cue = s.balls.find((b) => b.number === 0);
  const R = 0.028575;
  const suit = (n) => (n === 8 ? "eight" : n <= 7 ? "solid" : "stripe");
  const mine = s.groups.you;
  const onTable = s.balls.filter((b) => !b.pocketed && b.number !== 0);
  let targets = mine ? onTable.filter((b) => suit(b.number) === mine) : onTable.filter((b) => b.number !== 8);
  if (!targets.length) targets = onTable;

  let best = null;
  for (const t of targets) {
    for (const [pxp, pyp] of s.table.pockets) {
      const dx = pxp - t.x;
      const dy = pyp - t.y;
      const d = Math.hypot(dx, dy);
      const gx = t.x - (dx / d) * 2 * R;
      const gy = t.y - (dy / d) * 2 * R;
      const ax = gx - cue.x;
      const ay = gy - cue.y;
      const cut = Math.acos(
        Math.max(-1, Math.min(1, (ax * dx + ay * dy) / (Math.hypot(ax, ay) * d)))
      );
      // Nothing in the way of either leg.
      const clear = s.balls.every((b) => {
        if (b.pocketed || b.number === 0 || b.number === t.number) return true;
        for (const [x0, y0, x1, y1] of [
          [cue.x, cue.y, gx, gy],
          [t.x, t.y, pxp, pyp],
        ]) {
          const vx = x1 - x0;
          const vy = y1 - y0;
          const len2 = vx * vx + vy * vy;
          const u = Math.max(0, Math.min(1, ((b.x - x0) * vx + (b.y - y0) * vy) / len2));
          if (Math.hypot(b.x - (x0 + u * vx), b.y - (y0 + u * vy)) < 2 * R) return false;
        }
        return true;
      });
      if (!clear || cut > 1.1) continue;
      const score = cut + 0.25 * Math.hypot(ax, ay);
      if (!best || score < best.score) best = { score, angle: Math.atan2(ay, ax) };
    }
  }
  if (!best) return false;
  window.pocket.aim(best.angle);
  window.pocket.setPower(0.34);
  return true;
};

/**
 * Aim at whatever is legal, for when no pot is on.
 *
 * Without this the recording stops at the first snookered layout, which is
 * both common and early, and the clip becomes a break and one shot.
 */
const AIM_AT_ANY_LEGAL = () => {
  const s = window.pocket.state;
  const cue = s.balls.find((b) => b.number === 0);
  const suit = (n) => (n === 8 ? "eight" : n <= 7 ? "solid" : "stripe");
  const mine = s.groups.you;
  const onTable = s.balls.filter((b) => !b.pocketed && b.number !== 0);
  let targets = mine ? onTable.filter((b) => suit(b.number) === mine) : onTable;
  if (!targets.length) targets = onTable;
  if (!targets.length) return false;
  const near = targets.reduce((a, b) =>
    Math.hypot(b.x - cue.x, b.y - cue.y) < Math.hypot(a.x - cue.x, a.y - cue.y) ? b : a
  );
  window.pocket.aim(Math.atan2(near.y - cue.y, near.x - cue.x));
  window.pocket.setPower(0.3);
  return true;
};

/**
 * Aim down the longest clear line on the table.
 *
 * The inspector's whole point is the slip-to-roll handover landing on 5/7·v₀,
 * and it withdraws that prediction the moment the ball hits something first.
 * Whether a game shot happens to be clean is luck, so the still is taken after
 * a shot chosen to be one.
 */
const AIM_INTO_OPEN_SPACE = () => {
  const s = window.pocket.state;
  const cue = s.balls.find((b) => b.number === 0);
  const R = 0.028575;
  let best = null;
  for (let i = 0; i < 720; i++) {
    const angle = (i / 720) * 2 * Math.PI;
    const dx = Math.cos(angle);
    const dy = Math.sin(angle);
    // Distance to the cushion along this heading.
    let clear = Infinity;
    for (const [d, lo, hi] of [
      [dx, R - cue.x, s.table.length - R - cue.x],
      [dy, R - cue.y, s.table.width - R - cue.y],
    ]) {
      if (Math.abs(d) < 1e-9) continue;
      clear = Math.min(clear, (d > 0 ? hi : lo) / d);
    }
    for (const b of s.balls) {
      if (b.pocketed || b.number === 0) continue;
      const along = (b.x - cue.x) * dx + (b.y - cue.y) * dy;
      if (along <= 0) continue;
      const off = Math.abs((b.x - cue.x) * dy - (b.y - cue.y) * dx);
      if (off < 2 * R) clear = Math.min(clear, along);
    }
    // A heading down a pocket is disqualified rather than shortened: the cue
    // ball dropping ends the trace in the middle of the roll, which is a fine
    // thing for the panel to say and a poor thing for it to be a picture of.
    const scratches = s.table.pockets.some(([px, py]) => {
      const along = (px - cue.x) * dx + (py - cue.y) * dy;
      if (along <= 0) return false;
      const off = Math.abs((px - cue.x) * dy - (py - cue.y) * dx);
      return off < s.table.pocketRadius + R && along <= clear + s.table.pocketRadius;
    });
    if (scratches) continue;
    if (!best || clear > best.clear) best = { clear, angle };
  }
  if (!best) return 0;
  window.pocket.aim(best.angle);
  // A ball struck at v slides 12v²/(49 μ g) before it rolls, so the stroke is
  // sized to finish that inside the clear line with room to spare. Too hard and
  // it reaches a rail mid-slide, which withdraws the prediction just as surely
  // as hitting a ball does.
  const slideRoom = 0.55 * best.clear;
  const v = Math.sqrt((slideRoom * 49 * 0.2 * 9.81) / 12);
  window.pocket.setPower(Math.max(0.12, Math.min(0.6, v / 7.5)));
  return best.clear;
};

async function settle(page, timeout = 45000) {
  await page.waitForFunction(() => ["aim", "placing", "over"].includes(window.pocket.mode), {
    timeout,
    polling: 100,
  });
}

/** Wait for the balls to stop, which is earlier than waiting for the turn. */
async function settleShot(page, timeout = 45000) {
  await page.waitForFunction(() => !["rolling", "stroking"].includes(window.pocket.mode), {
    timeout,
    polling: 50,
  });
}

/**
 * Photograph the inspector showing a shot it can be held to.
 *
 * Two things make this more than a screenshot call. The panel traces whichever
 * ball was last struck, the bot's included, so the still has to be taken in the
 * gap between the cue ball stopping and the opponent replying. And the 5/7·v₀
 * line is only drawn when the shot stayed clean, which is the entire reason for
 * taking the picture — so the legend is read back before the shutter, and a
 * shot that touched something first buys another turn rather than a caption
 * explaining why the interesting line is missing.
 *
 * A cue ball that drops after rebounding satisfies the legend and still makes a
 * poor picture: the trace stops mid-roll and the panel is captioned with a
 * foul. The heading is chosen to miss the pockets, but only along the line to
 * the first rail, so the outcome is checked as well as the aim.
 */
async function captureInspector(page, panel, file) {
  const showsPrediction = () =>
    document.getElementById("trace-legend").textContent.includes("5/7·v₀ =");
  const cueStillOnTable = () => !window.pocket.state.balls.find((b) => b.number === 0).pocketed;

  for (let attempt = 0; attempt < 5; attempt++) {
    await settle(page);
    if (await page.evaluate(() => window.pocket.mode === "over")) break;
    await page.evaluate(() => {
      if (window.pocket.mode === "placing") {
        const s = window.pocket.state;
        window.pocket.place(s.table.length * 0.45, s.table.width * 0.5);
      }
    });
    // With no clean line available, play a pot instead: it rearranges the
    // table, which is what the next attempt needs.
    const clear = await page.evaluate(AIM_INTO_OPEN_SPACE);
    if (!clear && !(await page.evaluate(AIM_AT_BEST_POT))) break;
    await page.evaluate(() => window.pocket.shoot());
    await settleShot(page);
    if (await page.evaluate(showsPrediction) && (await page.evaluate(cueStillOnTable))) {
      await panel.screenshot({ path: file });
      return true;
    }
  }
  console.log("no clean slide-to-roll came up; the inspector still is of whatever was last hit");
  await panel.screenshot({ path: file });
  return false;
}

/**
 * Say how far the break opened the rack, and refuse a clip of one that did not.
 *
 * The first two seconds of the clip are the break, so a mis-struck one is the
 * loudest claim the documentation makes. It is also easy to make by accident
 * and hard to notice afterwards, since a rack that stays standing looks like a
 * physics limitation rather than a badly aimed cue.
 */
async function reportBreak(page, before) {
  const spread = await page.evaluate((prior) => {
    const live = window.pocket.state.balls.filter((b) => !b.pocketed && b.number !== 0);
    const cx = live.reduce((a, b) => a + b.x, 0) / live.length;
    const cy = live.reduce((a, b) => a + b.y, 0) / live.length;
    const moved = live.filter((b, i) => Math.hypot(b.x - prior[i][0], b.y - prior[i][1]) > 0.05);
    return {
      centroid: live.reduce((a, b) => a + Math.hypot(b.x - cx, b.y - cy), 0) / live.length,
      moved: moved.length,
      potted: 15 - live.length,
    };
  }, before);
  console.log(
    `break: ${spread.moved}/15 balls moved more than 5 cm, ` +
      `mean ${(spread.centroid * 100).toFixed(0)} cm from the pack centre, ` +
      `${spread.potted} potted`
  );
  // Displacement rather than the spread of the pack, because potting a ball
  // removes it from the pack and so *lowers* a spread measured over what is
  // left. Clipping the apex moves five balls; hitting it moves twelve.
  if (spread.moved < 10) {
    throw new Error(
      `the break only moved ${spread.moved} of 15 balls; a clip of that would ` +
        `misrepresent the simulator`
    );
  }
}

async function main() {
  const { url, out, seconds, noVideo } = parseArgs();

  let puppeteer;
  try {
    puppeteer = (await import("puppeteer-core")).default;
  } catch {
    skip("puppeteer-core is not installed (npm i puppeteer-core)");
  }
  const executablePath = CHROME_CANDIDATES.find((p) => existsSync(p));
  if (!executablePath) skip("no Chrome binary found; set CHROME_PATH");
  const haveFfmpeg = spawnSync("ffmpeg", ["-version"]).status === 0;

  mkdirSync(out, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath,
    headless: "new",
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--hide-scrollbars"],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1420, height: 940, deviceScaleFactor: 2 });
    await page.goto(url, { waitUntil: "networkidle0", timeout: 30000 });
    await page.waitForFunction(() => window.pocket !== undefined, { timeout: 10000 });
    await page.select("#difficulty", "sharp");
    // "Quick". The last seconds of a pool shot are balls creeping to a halt,
    // which is honest physics and dull footage; the page offers the speed, so
    // the recording uses it rather than editing the tails out afterwards.
    await page.select("#playback", "3");

    // Break, so the layout in the images is one the simulator produced. The
    // line is computed to the apex ball rather than guessed at: an earlier
    // hard-coded angle arrived 33 mm off it, and a break that clips the apex
    // leaves the rack standing, which made the clip an advertisement for a
    // problem the simulator does not have.
    await page.evaluate(() => {
      const s = window.pocket.state;
      window.pocket.place(s.table.length * 0.2, s.table.width * 0.52);
      const cue = s.balls.find((b) => b.number === 0);
      const apex = s.balls
        .filter((b) => !b.pocketed && b.number !== 0)
        .reduce((a, b) => (b.x < a.x ? b : a));
      // The same fraction off square the bot uses: dead centre sends the
      // energy back down the table instead of into the corners.
      window.pocket.aim(Math.atan2(apex.y - cue.y, apex.x - cue.x) + 0.004);
      window.pocket.setPower(0.95);
    });
    const before = await page.evaluate(() =>
      window.pocket.state.balls.filter((b) => b.number !== 0).map((b) => [b.x, b.y])
    );

    const frames = noVideo || !haveFfmpeg ? null : await startScreencast(page);
    await page.evaluate(() => window.pocket.shoot());
    await settle(page);
    await reportBreak(page, before);

    // Real shots, so the trace and the bot panel have content.
    for (let i = 0; i < 30; i++) {
      const mode = await page.evaluate(() => window.pocket.mode);
      if (mode === "over") break;
      if (mode === "placing") {
        await page.evaluate(() => {
          const s = window.pocket.state;
          window.pocket.place(s.table.length * (s.behindHeadString ? 0.2 : 0.45), s.table.width * 0.5);
        });
      }
      const aimed =
        (await page.evaluate(AIM_AT_BEST_POT)) || (await page.evaluate(AIM_AT_ANY_LEGAL));
      if (!aimed) break;
      await page.evaluate(() => window.pocket.shoot());
      await settle(page);
      if (frames && frames.elapsed() > seconds) break;
    }

    if (frames) {
      const { shots, marks } = await frames.stop();
      writeVideo(labelFrames(shots, marks), out);
    }

    // Line up a shot for the stills, then hold it.
    await page.evaluate(() => {
      if (window.pocket.mode === "placing") {
        const s = window.pocket.state;
        window.pocket.place(s.table.length * 0.45, s.table.width * 0.5);
      }
    });
    await page.evaluate(AIM_AT_BEST_POT);
    await new Promise((r) => setTimeout(r, 400));

    await (await page.$(".stage")).screenshot({ path: path.join(out, "web_game.png") });
    const panels = await page.$$("aside .panel");
    await panels[2].screenshot({ path: path.join(out, "web_bot.png") });

    await captureInspector(page, panels[1], path.join(out, "web_inspector.png"));

    console.log(`wrote stills to ${out}/`);
  } finally {
    await browser.close();
  }
}

/**
 * Capture through the DevTools screencast rather than by taking screenshots in
 * a loop: frames arrive as they are painted and carry their own timestamps, so
 * the clip runs at the speed the game actually ran at.
 */
async function startScreencast(page) {
  const client = await page.createCDPSession();
  const shots = [];
  const t0 = Date.now();
  client.on("Page.screencastFrame", async ({ data, metadata, sessionId }) => {
    shots.push({ data, t: metadata.timestamp });
    try {
      await client.send("Page.screencastFrameAck", { sessionId });
    } catch {
      /* the session closes while frames are still in flight */
    }
  });
  // What the page was doing when each frame was painted, sampled in the page so
  // it costs no round trips. Used to find the stretches where the bot is
  // searching and nothing on the table moves.
  await page.evaluate(() => {
    window.__modeMarks = [];
    window.__modeTimer = setInterval(() => {
      window.__modeMarks.push([performance.timeOrigin + performance.now(), window.pocket.mode]);
    }, 60);
  });
  await client.send("Page.startScreencast", {
    format: "jpeg",
    quality: 92,
    maxWidth: 1420,
    maxHeight: 940,
    everyNthFrame: 1,
  });
  return {
    elapsed: () => (Date.now() - t0) / 1000,
    async stop() {
      await client.send("Page.stopScreencast");
      await client.detach();
      const marks = await page.evaluate(() => {
        clearInterval(window.__modeTimer);
        return window.__modeMarks;
      });
      return { shots, marks };
    },
  };
}

/**
 * Tag each frame with what the page was doing when it was painted.
 *
 * Both clocks are milliseconds since the epoch — CDP's frame metadata in
 * seconds, the page's `timeOrigin + now()` in milliseconds — so they can be
 * merged by walking them together.
 */
function labelFrames(shots, marks) {
  let i = 0;
  return shots.map((shot) => {
    const at = shot.t * 1000;
    while (i + 1 < marks.length && marks[i + 1][0] <= at) i++;
    return { ...shot, mode: marks.length ? marks[i][1] : "unknown" };
  });
}

function writeVideo(shots, out) {
  if (shots.length < 10) {
    console.log(`only ${shots.length} frames captured; skipping the clip`);
    return;
  }
  const dir = path.join(out, ".frames");
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });

  // The bot searching is a still table with a progress bar creeping across it,
  // and at "sharp" it is most of the elapsed time. Left in, the clip is two
  // thirds nothing; cut out entirely, the game appears to play itself. So the
  // pause is kept and capped, and the README says the clip is trimmed there.
  let thinkingFor = 0;
  const kept = [];
  for (const [i, shot] of shots.entries()) {
    const next = shots[i + 1];
    const dt = next ? Math.min(0.5, Math.max(0.016, next.t - shot.t)) : 0.08;
    if (shot.mode === "thinking") {
      thinkingFor += dt;
      if (thinkingFor > THINKING_SECONDS) continue;
    } else {
      thinkingFor = 0;
    }
    kept.push({ data: shot.data, dt });
  }
  const trimmed = shots.length - kept.length;

  const lines = [];
  for (const [i, frame] of kept.entries()) {
    const name = `f${String(i).padStart(5, "0")}.jpg`;
    writeFileSync(path.join(dir, name), Buffer.from(frame.data, "base64"));
    lines.push(`file '${name}'`, `duration ${frame.dt.toFixed(4)}`);
  }
  lines.push(`file 'f${String(kept.length - 1).padStart(5, "0")}.jpg'`);
  const listFile = path.join(dir, "frames.txt");
  writeFileSync(listFile, lines.join("\n"));

  const mp4 = path.join(out, "web_demo.mp4");
  run("ffmpeg", [
    "-y", "-loglevel", "error",
    "-f", "concat", "-safe", "0", "-i", listFile,
    "-vf", "fps=30,scale=1100:-2:flags=lanczos",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart",
    mp4,
  ]);

  // A GIF as well: the README renders one inline everywhere, a video not
  // reliably. Trimmed, smaller and slower-sampled than the clip, because a
  // README that costs four megabytes to open is a README nobody scrolls.
  const palette = path.join(dir, "palette.png");
  const gifFilter = "fps=10,scale=680:-1:flags=lanczos";
  run("ffmpeg", [
    "-y", "-loglevel", "error", "-t", String(GIF_SECONDS), "-i", mp4,
    "-vf", `${gifFilter},palettegen=stats_mode=diff`,
    palette,
  ]);
  run("ffmpeg", [
    "-y", "-loglevel", "error", "-t", String(GIF_SECONDS), "-i", mp4, "-i", palette,
    "-lavfi", `${gifFilter}[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3`,
    path.join(out, "web_demo.gif"),
  ]);

  rmSync(dir, { recursive: true, force: true });
  console.log(
    `wrote ${kept.length} frames to ${mp4} and web_demo.gif ` +
      `(${trimmed} trimmed from the bot's searches)`
  );
}

function run(cmd, args) {
  const result = spawnSync(cmd, args, { stdio: "inherit" });
  if (result.status !== 0) throw new Error(`${cmd} exited ${result.status}`);
}

main().catch((error) => {
  console.error(`capture failed: ${error.message}`);
  process.exit(1);
});
