import * as THREE from 'three';
import { CinemaCamera } from '../camera/CinemaCamera.ts';
import { GpuParticleAdvect } from '../gpu/GpuParticleAdvect.ts';
import { getApproachState } from '../physics/approachOrbit.ts';
import { DT, T_HANDOFF, T_INITIAL_VIEW, T_MAX, Vec3 } from '../physics/constants.ts';
import { WorkerInMessage, WorkerOutMessage } from '../physics/nbodyWorker.ts';
import { CinemaRenderer } from '../render/CinemaRenderer.ts';

export const DEFAULT_EXPORT_START_GYR = T_INITIAL_VIEW;

export interface ExportConfig {
  width: number;
  height: number;
  fps: number;
  gyrPerSec: number;
  startGyr: number;
  testFrames: number | null;
}

export interface OfflineExportOptions {
  renderer: THREE.WebGLRenderer;
  canvas: HTMLCanvasElement;
  advect: GpuParticleAdvect;
  cinemaRenderer: CinemaRenderer;
  cinemaCamera: CinemaCamera;
  worker: Worker;
  config: ExportConfig;
}

function parsePositiveNumber(raw: string | null, fallback: number): number {
  if (raw === null) return fallback;
  const v = Number(raw);
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

function parsePositiveInt(raw: string | null, fallback: number): number {
  return Math.max(1, Math.round(parsePositiveNumber(raw, fallback)));
}

function parseStartGyr(raw: string | null, fallback: number): number {
  if (raw === null) return fallback;
  const v = Number(raw);
  return Number.isFinite(v) && v >= 0 && v < T_MAX ? v : fallback;
}

export function parseExportConfig(search: string): ExportConfig | null {
  const params = new URLSearchParams(search);
  if (params.get('export') !== '1') return null;

  const testRaw = params.get('testFrames');
  let testFrames: number | null = null;
  if (testRaw !== null) {
    const parsed = Number(testRaw);
    if (Number.isFinite(parsed) && parsed > 0) {
      testFrames = Math.round(parsed);
    }
  }

  return {
    width: parsePositiveInt(params.get('w'), 1920),
    height: parsePositiveInt(params.get('h'), 1080),
    fps: parsePositiveInt(params.get('fps'), 60),
    gyrPerSec: parsePositiveNumber(params.get('gyrPerSec'), 0.12),
    startGyr: parseStartGyr(params.get('startGyr'), DEFAULT_EXPORT_START_GYR),
    testFrames,
  };
}

function lerpVec3(a: Vec3, b: Vec3, alpha: number): Vec3 {
  const inv = 1.0 - alpha;
  return [
    a[0] * inv + b[0] * alpha,
    a[1] * inv + b[1] * alpha,
    a[2] * inv + b[2] * alpha,
  ];
}

export async function runOfflineExport(opts: OfflineExportOptions): Promise<void> {
  const { renderer, canvas, advect, cinemaRenderer, cinemaCamera, worker, config } = opts;

  try {
    renderer.setPixelRatio(1);
    renderer.setSize(config.width, config.height, false);
    canvas.style.width = `${config.width}px`;
    canvas.style.height = `${config.height}px`;
    cinemaCamera.setAspect(config.width / config.height);
    cinemaCamera.setDirectorMode(true);
    cinemaRenderer.resize();

    const gl = renderer.getContext() as WebGL2RenderingContext;
    if (gl.drawingBufferWidth !== config.width || gl.drawingBufferHeight !== config.height) {
      throw new Error(
        `Drawing buffer mismatch: expected ${config.width}x${config.height}, got ${gl.drawingBufferWidth}x${gl.drawingBufferHeight}`,
      );
    }

    const dtGyr = config.gyrPerSec / config.fps;
    const subSteps = Math.max(1, Math.round(DT / dtGyr));
    if (Math.abs(DT / subSteps - dtGyr) > 1e-6) {
      throw new Error(
        `dtGyr=${dtGyr} must divide DT=${DT} evenly (got subSteps=${subSteps}, DT/subSteps=${DT / subSteps})`,
      );
    }

    const approachFrames = Math.round(T_HANDOFF / dtGyr);
    if (Math.abs(approachFrames * dtGyr - T_HANDOFF) > 1e-6) {
      throw new Error(`dtGyr=${dtGyr} must divide T_HANDOFF=${T_HANDOFF} evenly`);
    }

    const maxNbodySteps = Math.round((T_MAX - T_HANDOFF) / DT);
    let startApproachFrame = approachFrames;
    let startStep = 0;
    let fullTotalFrames: number;

    if (config.startGyr >= T_HANDOFF) {
      startStep = Math.round((config.startGyr - T_HANDOFF) / DT);
      if (Math.abs(T_HANDOFF + startStep * DT - config.startGyr) > 1e-6) {
        throw new Error(`startGyr=${config.startGyr} must align to an N-body step of DT=${DT}`);
      }
      fullTotalFrames = 1 + (maxNbodySteps - startStep) * subSteps;
    } else {
      startApproachFrame = Math.round(config.startGyr / dtGyr);
      if (Math.abs(startApproachFrame * dtGyr - config.startGyr) > 1e-6) {
        throw new Error(`startGyr=${config.startGyr} must align to dtGyr=${dtGyr}`);
      }
      fullTotalFrames = (approachFrames - startApproachFrame) + 1 + maxNbodySteps * subSteps;
    }

    const targetFrames = config.testFrames
      ? Math.min(config.testFrames, fullTotalFrames)
      : fullTotalFrames;

    let pendingResolve: ((msg: WorkerOutMessage) => void) | null = null;
    let pendingReject: ((err: Error) => void) | null = null;

    worker.onmessage = (ev: MessageEvent<WorkerOutMessage>) => {
      const resolve = pendingResolve;
      pendingResolve = null;
      pendingReject = null;
      if (resolve) resolve(ev.data);
    };

    worker.onerror = (ev: ErrorEvent) => {
      const reject = pendingReject;
      pendingResolve = null;
      pendingReject = null;
      if (reject) reject(new Error(`Physics worker error: ${ev.message || 'unknown error'}`));
    };

    const callWorker = <T extends WorkerOutMessage['type']>(
      msg: WorkerInMessage,
      expectedType: T,
    ): Promise<Extract<WorkerOutMessage, { type: T }>> => {
      if (pendingResolve) {
        return Promise.reject(new Error('Concurrent worker call is not allowed in OfflineExporter'));
      }
      return new Promise((resolve, reject) => {
        pendingResolve = (out) => {
          if (out.type !== expectedType) {
            reject(new Error(`Unexpected worker response type "${out.type}", expected "${expectedType}"`));
            return;
          }
          resolve(out as Extract<WorkerOutMessage, { type: T }>);
        };
        pendingReject = reject;
        worker.postMessage(msg);
      });
    };

    const initWorker = () => callWorker({ type: 'init' }, 'inited');
    const stepWorker = (id: number) => callWorker({ type: 'step', id }, 'stepped');

    const inited = await initWorker();
    advect.uploadOctree(inited.treePosMass, inited.treeMeta, inited.nodeCount);

    const pixelBuf = new Uint8Array(config.width * config.height * 4);
    const captureAndSend = async (frameIdx: number, timeGyr: number): Promise<void> => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.readPixels(0, 0, config.width, config.height, gl.RGBA, gl.UNSIGNED_BYTE, pixelBuf);

      const res = await fetch('/frame', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
          'X-Frame-Index': String(frameIdx),
          'X-Total-Frames': String(targetFrames),
          'X-Time-Gyr': timeGyr.toFixed(4),
        },
        body: pixelBuf,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`POST /frame failed (${res.status}): ${text}`);
      }
    };

    let frameIdx = 0;
    const frameDtSec = 1.0 / config.fps;

    // Phase 1: Analytic RK4 Approach (when startGyr < T_HANDOFF)
    for (let f = startApproachFrame; f < approachFrames && frameIdx < targetFrames; f++) {
      const t = f * dtGyr;
      const st = getApproachState(t);
      advect.setApproachState(t, st.mwPos, st.mwVel, st.m31Pos, st.m31Vel);

      if (frameIdx === 0) {
        cinemaCamera.snapToDirector(t, st.mwPos, st.m31Pos, st.separationKpc, 0.0);
      } else {
        cinemaCamera.update(
          frameDtSec,
          t,
          st.mwPos,
          st.m31Pos,
          st.separationKpc,
          frameIdx * frameDtSec,
          false,
        );
      }

      cinemaRenderer.render(cinemaCamera.camera, st.mwPos, st.m31Pos, st.separationKpc);
      await captureAndSend(frameIdx, t);
      frameIdx++;
    }

    // Phase 2: Full Barnes-Hut + GPU N-Body (T = T_HANDOFF -> T_MAX)
    if (frameIdx < targetFrames) {
      advect.resetToHandoff();
      let gpuIsHalfShifted = false;
      let prev = {
        timeGyr: T_HANDOFF,
        mwCenter: inited.mwCenter,
        m31Center: inited.m31Center,
        sepKpc: inited.sepKpc,
      };

      for (let step = 1; step <= startStep; step++) {
        const msg = await stepWorker(step);
        const kickDt = gpuIsHalfShifted ? DT : 0.5 * DT;
        advect.step(kickDt, DT, DT, !gpuIsHalfShifted);
        gpuIsHalfShifted = true;
        advect.uploadOctree(msg.treePosMass, msg.treeMeta, msg.nodeCount);
        prev = {
          timeGyr: msg.timeGyr,
          mwCenter: msg.mwCenter,
          m31Center: msg.m31Center,
          sepKpc: msg.sepKpc,
        };
      }

      if (frameIdx === 0) {
        cinemaCamera.snapToDirector(prev.timeGyr, prev.mwCenter, prev.m31Center, prev.sepKpc, 0.0);
      } else {
        cinemaCamera.update(
          frameDtSec,
          prev.timeGyr,
          prev.mwCenter,
          prev.m31Center,
          prev.sepKpc,
          frameIdx * frameDtSec,
          false,
        );
      }
      cinemaRenderer.render(cinemaCamera.camera, prev.mwCenter, prev.m31Center, prev.sepKpc);
      await captureAndSend(frameIdx, prev.timeGyr);
      frameIdx++;

      for (let step = startStep + 1; step <= maxNbodySteps && frameIdx < targetFrames; step++) {
        const msg = await stepWorker(step);
        const kickDt = gpuIsHalfShifted ? DT : 0.5 * DT;
        advect.step(kickDt, DT, DT, !gpuIsHalfShifted);
        gpuIsHalfShifted = true;
        advect.uploadOctree(msg.treePosMass, msg.treeMeta, msg.nodeCount);

        for (let sub = 1; sub <= subSteps && frameIdx < targetFrames; sub++) {
          const alpha = sub / subSteps;
          const tInterp = prev.timeGyr * (1.0 - alpha) + msg.timeGyr * alpha;
          const mwInterp = lerpVec3(prev.mwCenter, msg.mwCenter, alpha);
          const m31Interp = lerpVec3(prev.m31Center, msg.m31Center, alpha);
          const sepInterp = prev.sepKpc * (1.0 - alpha) + msg.sepKpc * alpha;

          advect.setInterpolatedPos(alpha);
          cinemaCamera.update(
            frameDtSec,
            tInterp,
            mwInterp,
            m31Interp,
            sepInterp,
            frameIdx * frameDtSec,
            false,
          );
          cinemaRenderer.render(cinemaCamera.camera, mwInterp, m31Interp, sepInterp);
          await captureAndSend(frameIdx, tInterp);
          frameIdx++;
        }

        prev = {
          timeGyr: msg.timeGyr,
          mwCenter: msg.mwCenter,
          m31Center: msg.m31Center,
          sepKpc: msg.sepKpc,
        };
      }
    }

    const doneRes = await fetch('/done', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: frameIdx }),
    });
    if (!doneRes.ok) {
      throw new Error(`POST /done failed (${doneRes.status})`);
    }
  } catch (err) {
    const message = err instanceof Error ? (err.stack ?? err.message) : String(err);
    await fetch('/error', {
      method: 'POST',
      headers: { 'Content-Type': 'text/plain' },
      body: message,
    }).catch(() => {});
  }
}
