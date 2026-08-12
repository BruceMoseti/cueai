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

const GIF_SECONDS = 12;

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, fallback) => {
    const i = args.indexOf(flag);
    return i >= 0 ? args[i + 1] : fallback;
  };
  return {
    url: get("--url", "http://localhost:8123/index.html"),
    out: get("--out", "docs/assets"),
    seconds: Number(get("--seconds", 16)),
    noVideo: args.includes("--no-video"),
  };
}

function skip(reason) {
  console.log(`capture skipped: ${reason}`);
  process.exit(0);
}

/** Aim at the ghost ball for the straightest clear pot, the way a player would. */
const AIM_AT_BEST_POT = () => {
  const s = window.cueai.state;
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
  window.cueai.aim(best.angle);
  window.cueai.setPower(0.34);
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
  const s = window.cueai.state;
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
    if (!best || clear > best.clear) best = { clear, angle };
  }
  window.cueai.aim(best.angle);
  // Enough to slide for a visible fraction of a second, not enough to reach
  // the far rail before it starts rolling.
  window.cueai.setPower(0.32);
  return best.clear;
};

async function settle(page, timeout = 45000) {
  await page.waitForFunction(() => ["aim", "placing", "over"].includes(window.cueai.mode), {
    timeout,
    polling: 100,
  });
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
    await page.waitForFunction(() => window.cueai !== undefined, { timeout: 10000 });
    await page.select("#difficulty", "sharp");
    await page.select("#playback", "1.8");

    // Break, so the layout in the images is one the simulator produced.
    await page.evaluate(() => {
      const s = window.cueai.state;
      window.cueai.place(s.table.length * 0.2, s.table.width * 0.52);
      window.cueai.aim(0.006);
      window.cueai.setPower(0.95);
    });

    const frames = noVideo || !haveFfmpeg ? null : await startScreencast(page);
    await page.evaluate(() => window.cueai.shoot());
    await settle(page);

    // A couple of real shots, so the trace and the bot panel have content.
    for (let i = 0; i < 4; i++) {
      const mode = await page.evaluate(() => window.cueai.mode);
      if (mode === "over") break;
      if (mode === "placing") {
        await page.evaluate(() => {
          const s = window.cueai.state;
          window.cueai.place(s.table.length * (s.behindHeadString ? 0.2 : 0.45), s.table.width * 0.5);
        });
      }
      const aimed = await page.evaluate(AIM_AT_BEST_POT);
      if (!aimed) break;
      await page.evaluate(() => window.cueai.shoot());
      await settle(page);
      if (frames && frames.elapsed() > seconds) break;
    }

    if (frames) {
      const clip = await frames.stop();
      writeVideo(clip, out);
    }

    // Line up a shot for the stills, then hold it.
    await page.evaluate(() => {
      if (window.cueai.mode === "placing") {
        const s = window.cueai.state;
        window.cueai.place(s.table.length * 0.45, s.table.width * 0.5);
      }
    });
    await page.evaluate(AIM_AT_BEST_POT);
    await new Promise((r) => setTimeout(r, 400));

    await (await page.$(".stage")).screenshot({ path: path.join(out, "web_game.png") });
    await (await page.$(".table-shell")).screenshot({ path: path.join(out, "web_table.png") });
    const panels = await page.$$("aside .panel");
    await panels[2].screenshot({ path: path.join(out, "web_bot.png") });

    // The inspector still gets a shot picked for it: see AIM_INTO_OPEN_SPACE.
    if (await page.evaluate(() => window.cueai.mode === "aim")) {
      await page.evaluate(AIM_INTO_OPEN_SPACE);
      await page.evaluate(() => window.cueai.shoot());
      await settle(page);
    }
    await panels[1].screenshot({ path: path.join(out, "web_inspector.png") });
    await page.evaluate(() => document.getElementById("how").scrollIntoView());
    await new Promise((r) => setTimeout(r, 250));
    await (await page.$("#how")).screenshot({ path: path.join(out, "web_explainer.png") });

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
      return shots;
    },
  };
}

function writeVideo(shots, out) {
  if (shots.length < 10) {
    console.log(`only ${shots.length} frames captured; skipping the clip`);
    return;
  }
  const dir = path.join(out, ".frames");
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });

  const lines = [];
  for (const [i, shot] of shots.entries()) {
    const name = `f${String(i).padStart(5, "0")}.jpg`;
    writeFileSync(path.join(dir, name), Buffer.from(shot.data, "base64"));
    const next = shots[i + 1];
    const dt = next ? Math.min(0.5, Math.max(0.016, next.t - shot.t)) : 0.08;
    lines.push(`file '${name}'`, `duration ${dt.toFixed(4)}`);
  }
  lines.push(`file 'f${String(shots.length - 1).padStart(5, "0")}.jpg'`);
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
  const gifFilter = "fps=12,scale=760:-1:flags=lanczos";
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
  console.log(`wrote ${shots.length} frames to ${mp4} and web_demo.gif`);
}

function run(cmd, args) {
  const result = spawnSync(cmd, args, { stdio: "inherit" });
  if (result.status !== 0) throw new Error(`${cmd} exited ${result.status}`);
}

main().catch((error) => {
  console.error(`capture failed: ${error.message}`);
  process.exit(1);
});
