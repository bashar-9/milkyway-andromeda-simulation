import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const DIST_DIR = path.join(ROOT_DIR, 'dist');

const DT = 0.004;
const T_HANDOFF = 2.60;
const T_MAX = 7.5;
const DEFAULT_START_GYR = 2.84;

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
};

function parseArgs(argv) {
  const opts = {
    out: 'milky-way-andromeda-collision.mp4',
    width: 1920,
    height: 1080,
    fps: 60,
    gyrPerSec: 0.12,
    startGyr: DEFAULT_START_GYR,
    crf: 15,
    preset: 'slow',
    testFrames: null,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const eqIdx = arg.indexOf('=');
    const key = eqIdx >= 0 ? arg.slice(2, eqIdx) : arg.slice(2);
    const val = eqIdx >= 0 ? arg.slice(eqIdx + 1) : argv[++i];
    if (val === undefined) {
      throw new Error(`Missing value for --${key}`);
    }

    switch (key) {
      case 'out':
        opts.out = val;
        break;
      case 'width':
        opts.width = parseInt(val, 10);
        break;
      case 'height':
        opts.height = parseInt(val, 10);
        break;
      case 'fps':
        opts.fps = parseInt(val, 10);
        break;
      case 'gyr-per-sec':
        opts.gyrPerSec = parseFloat(val);
        break;
      case 'start-gyr':
        opts.startGyr = parseFloat(val);
        break;
      case 'crf':
        opts.crf = parseInt(val, 10);
        break;
      case 'preset':
        opts.preset = val;
        break;
      case 'test-frames':
        opts.testFrames = parseInt(val, 10);
        break;
      default:
        throw new Error(`Unknown option: --${key}`);
    }
  }

  if (!Number.isInteger(opts.width) || opts.width <= 0) throw new Error(`Invalid --width: ${opts.width}`);
  if (!Number.isInteger(opts.height) || opts.height <= 0) throw new Error(`Invalid --height: ${opts.height}`);
  if (!Number.isInteger(opts.fps) || opts.fps <= 0) throw new Error(`Invalid --fps: ${opts.fps}`);
  if (!Number.isFinite(opts.gyrPerSec) || opts.gyrPerSec <= 0) {
    throw new Error(`Invalid --gyr-per-sec: ${opts.gyrPerSec}`);
  }
  if (!Number.isFinite(opts.startGyr) || opts.startGyr < 0 || opts.startGyr >= T_MAX) {
    throw new Error(`Invalid --start-gyr: ${opts.startGyr}`);
  }
  if (!Number.isInteger(opts.crf) || opts.crf < 0 || opts.crf > 51) {
    throw new Error(`Invalid --crf: ${opts.crf}`);
  }
  if (opts.testFrames !== null && (!Number.isInteger(opts.testFrames) || opts.testFrames <= 0)) {
    throw new Error(`Invalid --test-frames: ${opts.testFrames}`);
  }

  return opts;
}

function findBrowserExecutable() {
  const candidates = [
    process.env.CHROME_PATH,
    // Windows
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
    // macOS
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    // Linux
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/bin/microsoft-edge',
    '/snap/bin/chromium',
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(
    `Could not locate Chrome, Chromium, or Edge executable (set CHROME_PATH to override). Checked:\n${candidates.join('\n')}`,
  );
}

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function formatDurationSec(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '--:--';
  const whole = Math.round(sec);
  const mins = Math.floor(whole / 60);
  const rem = whole % 60;
  return `${mins}m${String(rem).padStart(2, '0')}s`;
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const distIndex = path.join(DIST_DIR, 'index.html');
  if (!fs.existsSync(distIndex)) {
    console.error(`Missing build output at ${distIndex}. Please run "npm run build" first.`);
    process.exit(1);
  }

  const browserPath = findBrowserExecutable();
  const outPath = path.isAbsolute(opts.out) ? opts.out : path.resolve(ROOT_DIR, opts.out);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });

  const dtGyr = opts.gyrPerSec / opts.fps;
  const subSteps = Math.max(1, Math.round(DT / dtGyr));
  if (Math.abs(DT / subSteps - dtGyr) > 1e-6) {
    throw new Error(
      `gyr-per-sec / fps (${dtGyr}) must divide DT=${DT} evenly (got subSteps=${subSteps})`,
    );
  }

  const approachFrames = Math.round(T_HANDOFF / dtGyr);
  if (Math.abs(approachFrames * dtGyr - T_HANDOFF) > 1e-6) {
    throw new Error(`dtGyr=${dtGyr} must divide T_HANDOFF=${T_HANDOFF} evenly`);
  }

  const maxNbodySteps = Math.round((T_MAX - T_HANDOFF) / DT);
  let fullTotalFrames;
  if (opts.startGyr >= T_HANDOFF) {
    const startStep = Math.round((opts.startGyr - T_HANDOFF) / DT);
    if (Math.abs(T_HANDOFF + startStep * DT - opts.startGyr) > 1e-6) {
      throw new Error(`start-gyr (${opts.startGyr}) must align to DT=${DT}`);
    }
    fullTotalFrames = 1 + (maxNbodySteps - startStep) * subSteps;
  } else {
    const startApproachFrame = Math.round(opts.startGyr / dtGyr);
    if (Math.abs(startApproachFrame * dtGyr - opts.startGyr) > 1e-6) {
      throw new Error(`start-gyr (${opts.startGyr}) must align to dtGyr=${dtGyr}`);
    }
    fullTotalFrames = (approachFrames - startApproachFrame) + 1 + maxNbodySteps * subSteps;
  }
  const totalFrames = opts.testFrames ? Math.min(opts.testFrames, fullTotalFrames) : fullTotalFrames;
  const expectedBytesPerFrame = opts.width * opts.height * 4;

  let vf;
  if (totalFrames >= opts.fps * 4) {
    const fadeOutStart = totalFrames / opts.fps - 1.4;
    vf = `vflip,fade=t=in:st=0:d=0.75,fade=t=out:st=${fadeOutStart.toFixed(3)}:d=1.4,scale=out_color_matrix=bt709:out_range=tv`;
  } else {
    vf = 'vflip,scale=out_color_matrix=bt709:out_range=tv';
  }

  const tmpProfileDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mw-m31-export-'));
  console.log(
    `[export] Target: ${opts.width}x${opts.height} @ ${opts.fps} fps | ${totalFrames} frames (${(totalFrames / opts.fps).toFixed(2)}s) -> ${outPath}`,
  );
  console.log(`[export] Browser: ${browserPath}`);
  console.log(`[export] FFmpeg preset=${opts.preset} crf=${opts.crf}`);

  const ffmpegArgs = [
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'rgba',
    '-s', `${opts.width}x${opts.height}`,
    '-r', String(opts.fps),
    '-i', 'pipe:0',
    '-vf', vf,
    '-c:v', 'libx264',
    '-preset', opts.preset,
    '-crf', String(opts.crf),
    '-pix_fmt', 'yuv420p',
    '-profile:v', 'high',
    '-level', '4.2',
    '-x264-params', 'aq-mode=3:aq-strength=0.85:deblock=-1,-1',
    '-color_range', 'tv',
    '-colorspace', 'bt709',
    '-color_primaries', 'bt709',
    '-color_trc', 'bt709',
    '-movflags', '+faststart',
    outPath,
  ];

  const ffmpegProc = spawn('ffmpeg', ffmpegArgs, {
    stdio: ['pipe', 'ignore', 'pipe'],
  });

  const ffmpegStderrLines = [];
  ffmpegProc.stderr.on('data', (chunk) => {
    const text = chunk.toString('utf8');
    for (const line of text.split(/\r?\n/)) {
      if (!line.trim()) continue;
      ffmpegStderrLines.push(line);
      if (ffmpegStderrLines.length > 60) ffmpegStderrLines.shift();
    }
  });

  const ffmpegExitPromise = new Promise((resolve, reject) => {
    ffmpegProc.on('error', (err) => reject(err));
    ffmpegProc.on('close', (code, signal) => {
      resolve({ code, signal });
    });
  });

  let chromeProc = null;
  let finishing = false;
  let failed = false;
  let receivedFrames = 0;
  let firstFrameAtMs = null;
  const startWallMs = Date.now();

  const cleanup = () => {
    if (chromeProc && !chromeProc.killed) {
      try {
        chromeProc.kill('SIGKILL');
      } catch {
        // Ignore cleanup errors
      }
    }
    try {
      fs.rmSync(tmpProfileDir, { recursive: true, force: true });
    } catch {
      // Windows Chrome may hold a brief handle; ignore
    }
  };

  const abortWithError = (err) => {
    if (failed) return;
    failed = true;
    console.error('[export] ERROR:', err instanceof Error ? err.stack || err.message : String(err));
    if (ffmpegStderrLines.length > 0) {
      console.error('[export] Last ffmpeg output:\n' + ffmpegStderrLines.slice(-20).join('\n'));
    }
    if (!ffmpegProc.killed) {
      try {
        ffmpegProc.stdin.destroy();
        ffmpegProc.kill('SIGKILL');
      } catch {
        // Ignore
      }
    }
    cleanup();
    server.close();
    process.exit(1);
  };

  ffmpegExitPromise
    .then(({ code, signal }) => {
      if (!finishing && !failed) {
        abortWithError(new Error(`FFmpeg exited unexpectedly (code=${code}, signal=${signal})`));
      }
    })
    .catch((err) => abortWithError(err));

  const server = http.createServer(async (req, res) => {
    try {
      const reqUrl = new URL(req.url || '/', 'http://127.0.0.1');

      if (req.method === 'POST' && reqUrl.pathname === '/frame') {
        const buf = await readRequestBody(req);
        if (buf.length !== expectedBytesPerFrame) {
          throw new Error(
            `Frame byte length mismatch: expected ${expectedBytesPerFrame}, got ${buf.length}`,
          );
        }

        const frameIdxHeader = Number(req.headers['x-frame-index'] ?? receivedFrames);
        const totalHeader = Number(req.headers['x-total-frames'] ?? totalFrames);
        const timeGyrHeader = req.headers['x-time-gyr'] ?? '?';

        if (!ffmpegProc.stdin.write(buf)) {
          await once(ffmpegProc.stdin, 'drain');
        }

        receivedFrames++;
        if (firstFrameAtMs === null) {
          firstFrameAtMs = Date.now();
        }

        const frameNum = frameIdxHeader + 1;
        if (frameNum === 1 || frameNum % 100 === 0 || frameNum === totalHeader) {
          const elapsedSec = Math.max(0.001, (Date.now() - firstFrameAtMs) / 1000);
          const fps = receivedFrames / elapsedSec;
          const remaining = Math.max(0, totalHeader - frameNum);
          const etaSec = fps > 0 ? remaining / fps : -1;
          const pct = ((frameNum / Math.max(1, totalHeader)) * 100).toFixed(1);
          console.log(
            `[export] Frame ${frameNum}/${totalHeader} (${pct}%) | T+${timeGyrHeader} Gyr | ${fps.toFixed(2)} fps | ETA ${formatDurationSec(etaSec)}`,
          );
        }

        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('ok');
        return;
      }

      if (req.method === 'POST' && reqUrl.pathname === '/done') {
        const body = await readRequestBody(req);
        let declaredFrames = receivedFrames;
        try {
          const parsed = JSON.parse(body.toString('utf8'));
          if (typeof parsed.frames === 'number') declaredFrames = parsed.frames;
        } catch {
          // Ignore malformed optional JSON
        }

        finishing = true;
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('ok');

        console.log(`[export] Finalizing MP4 encoding (${declaredFrames} frames)...`);
        ffmpegProc.stdin.end();
        const { code, signal } = await ffmpegExitPromise;
        if (code !== 0) {
          throw new Error(`FFmpeg exited with status ${code} (signal=${signal})`);
        }

        cleanup();
        server.close();

        const totalElapsedSec = (Date.now() - startWallMs) / 1000;
        const sizeMb = (fs.statSync(outPath).size / (1024 * 1024)).toFixed(2);
        console.log(
          `[export] Complete: ${declaredFrames} frames in ${formatDurationSec(totalElapsedSec)} (${(declaredFrames / Math.max(0.001, totalElapsedSec)).toFixed(2)} avg fps) | ${outPath} (${sizeMb} MB)`,
        );
        process.exit(0);
      }

      if (req.method === 'POST' && reqUrl.pathname === '/error') {
        const body = await readRequestBody(req);
        abortWithError(new Error(`Renderer reported error:\n${body.toString('utf8')}`));
        return;
      }

      if (req.method === 'GET') {
        let relPath = decodeURIComponent(reqUrl.pathname);
        if (relPath === '/') relPath = '/index.html';
        console.log(`[export] GET ${relPath}`);
        const safePath = path.resolve(DIST_DIR, '.' + relPath);
        if (!safePath.startsWith(DIST_DIR)) {
          res.writeHead(403);
          res.end('Forbidden');
          return;
        }
        if (!fs.existsSync(safePath) || fs.statSync(safePath).isDirectory()) {
          res.writeHead(404);
          res.end('Not Found');
          return;
        }

        const ext = path.extname(safePath).toLowerCase();
        const mime = MIME_TYPES[ext] || 'application/octet-stream';
        res.writeHead(200, { 'Content-Type': mime });
        fs.createReadStream(safePath).pipe(res);
        return;
      }

      res.writeHead(405);
      res.end('Method Not Allowed');
    } catch (err) {
      res.writeHead(500);
      res.end('Internal Server Error');
      abortWithError(err);
    }
  });

  server.requestTimeout = 0;
  server.headersTimeout = 0;

  process.on('SIGINT', () => abortWithError(new Error('Interrupted by SIGINT')));
  process.on('SIGTERM', () => abortWithError(new Error('Interrupted by SIGTERM')));

  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const addr = server.address();
  if (!addr || typeof addr === 'string') {
    throw new Error('Failed to bind local export HTTP server');
  }

  const query = new URLSearchParams({
    export: '1',
    w: String(opts.width),
    h: String(opts.height),
    fps: String(opts.fps),
    gyrPerSec: String(opts.gyrPerSec),
    startGyr: String(opts.startGyr),
  });
  if (opts.testFrames !== null) {
    query.set('testFrames', String(opts.testFrames));
  }
  const exportUrl = `http://127.0.0.1:${addr.port}/?${query.toString()}`;

  const chromeArgs = [
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu-sandbox',
    '--enable-webgl',
    '--ignore-gpu-blocklist',
    `--window-size=${opts.width},${opts.height}`,
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
    `--user-data-dir=${tmpProfileDir}`,
    exportUrl,
  ];

  chromeProc = spawn(browserPath, chromeArgs, {
    stdio: ['ignore', 'ignore', 'pipe'],
  });

  const chromeStderr = [];
  chromeProc.stderr.on('data', (chunk) => {
    const text = chunk.toString('utf8');
    for (const line of text.split(/\r?\n/)) {
      if (!line.trim()) continue;
      chromeStderr.push(line);
      if (chromeStderr.length > 40) chromeStderr.shift();
    }
  });

  const firstFrameTimer = setTimeout(() => {
    if (receivedFrames === 0 && !finishing && !failed) {
      const tail = chromeStderr.length > 0 ? `\nChrome stderr:\n${chromeStderr.slice(-20).join('\n')}` : '';
      abortWithError(new Error(`Timed out waiting 30s for first rendered frame${tail}`));
    }
  }, 30000);
  firstFrameTimer.unref();

  chromeProc.on('error', (err) => abortWithError(err));
  chromeProc.on('close', (code, signal) => {
    clearTimeout(firstFrameTimer);
    if (!finishing && !failed) {
      const tail = chromeStderr.length > 0 ? `\nChrome stderr:\n${chromeStderr.slice(-15).join('\n')}` : '';
      abortWithError(
        new Error(`Headless browser exited early (code=${code}, signal=${signal}) after ${receivedFrames} frames${tail}`),
      );
    }
  });
}

main().catch((err) => {
  console.error('[export] Fatal:', err instanceof Error ? err.stack || err.message : String(err));
  process.exit(1);
});
