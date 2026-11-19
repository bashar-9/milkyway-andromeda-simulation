import * as THREE from 'three';
import { CinemaCamera } from './camera/CinemaCamera.ts';
import { parseExportConfig, runOfflineExport } from './export/OfflineExporter.ts';
import { GpuParticleAdvect } from './gpu/GpuParticleAdvect.ts';
import { generateVisualParticles } from './gpu/visualParticles.ts';
import { getApproachState } from './physics/approachOrbit.ts';
import {
  BASIS_M31_VIEW,
  BASIS_MW_VIEW,
  CHECKPOINT_SLICE_COUNT,
  CHECKPOINT_STEP_STRIDE,
  DT,
  M31_BULGE_COUNT,
  M31_DISK_COUNT,
  M31_HALO_COUNT,
  M31_PARAMS,
  MW_BULGE_COUNT,
  MW_DISK_COUNT,
  MW_HALO_COUNT,
  MW_PARAMS,
  T_HANDOFF,
  T_INITIAL_VIEW,
  T_MAX,
  Vec3,
} from './physics/constants.ts';
import { buildGalaxyRadialProfile, sampleMacroGalaxy } from './physics/galaxyModel.ts';
import { WorkerInMessage, WorkerOutMessage } from './physics/nbodyWorker.ts';
import { createRng } from './physics/random.ts';
import { CinemaRenderer } from './render/CinemaRenderer.ts';
import './style.css';
import { MinimalHud } from './ui/MinimalHud.ts';

type SimMode = 'boot_preroll' | 'approach' | 'nbody' | 'seeking';

const SPEEDS = [0.5, 1, 2, 4] as const;
const BASE_GYR_PER_SEC = 0.10;
const MAX_NBODY_STEPS = Math.round((T_MAX - T_HANDOFF) / DT);
const BOOT_TARGET_STEP = Math.round((T_INITIAL_VIEW - T_HANDOFF) / DT);

function bootstrap(): void {
  const canvas = document.getElementById('sim-canvas') as HTMLCanvasElement | null;
  const hudRoot = document.getElementById('hud-root') as HTMLElement | null;
  if (!canvas || !hudRoot) throw new Error('Missing #sim-canvas or #hud-root');

  const exportCfg = parseExportConfig(window.location.search);

  if (exportCfg) {
    const reportEarlyError = (err: unknown): void => {
      const message = err instanceof Error ? (err.stack ?? err.message) : String(err);
      void fetch('/error', {
        method: 'POST',
        headers: { 'Content-Type': 'text/plain' },
        body: message,
      }).catch(() => {});
    };
    window.addEventListener('error', (e) => reportEarlyError(e.error ?? e.message));
    window.addEventListener('unhandledrejection', (e) => reportEarlyError(e.reason));
  }

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    powerPreference: 'high-performance',
    alpha: false,
    preserveDrawingBuffer: exportCfg !== null,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);

  if (!renderer.capabilities.isWebGL2) {
    throw new Error('WebGL2 is required for this simulation');
  }

  // Build exact equilibrium galaxy profiles (matching nbodyWorker seed & parameters)
  const mwAnalytic = buildGalaxyRadialProfile(MW_PARAMS);
  const m31Analytic = buildGalaxyRadialProfile(M31_PARAMS);
  const handoff = getApproachState(T_HANDOFF);
  const initRng = createRng(421337);
  const mwMacro = sampleMacroGalaxy(
    mwAnalytic,
    MW_HALO_COUNT,
    MW_DISK_COUNT,
    MW_BULGE_COUNT,
    BASIS_MW_VIEW,
    handoff.mwPos,
    handoff.mwVel,
    initRng,
  );
  const m31Macro = sampleMacroGalaxy(
    m31Analytic,
    M31_HALO_COUNT,
    M31_DISK_COUNT,
    M31_BULGE_COUNT,
    BASIS_M31_VIEW,
    handoff.m31Pos,
    handoff.m31Vel,
    initRng,
  );

  const visualData = generateVisualParticles(
    mwMacro.eqProfile,
    m31Macro.eqProfile,
    handoff.mwPos,
    handoff.mwVel,
    handoff.m31Pos,
    handoff.m31Vel,
  );

  const advect = new GpuParticleAdvect(renderer, visualData);
  const cinemaRenderer = new CinemaRenderer(renderer, advect);

  // Simulation & playback state
  let simMode: SimMode = 'boot_preroll';
  let playing = true;
  let speedIdx = 1; // default 1x
  let currentTimeGyr = T_HANDOFF;
  let nbodyStepIndex = 0;
  let gpuIsHalfShifted = false;
  let nbodyTimeDebt = 0;

  let workerReady = false;
  let workerInFlight = false;
  let stepReqId = 0;

  let seekTargetStep: number | null = BOOT_TARGET_STEP;
  let pendingRestoreSlice: number | null = null;
  let maxCachedTimeGyr = T_HANDOFF;
  const cachedSlices = new Map<number, number>(); // slice -> stepIndex

  let mwCenter: Vec3 = [...handoff.mwPos];
  let m31Center: Vec3 = [...handoff.m31Pos];
  let sepKpc = handoff.separationKpc;
  let relSpeedKms = handoff.relSpeedKms;

  const cinemaCamera = new CinemaCamera(
    canvas,
    window.innerWidth / Math.max(1, window.innerHeight),
    exportCfg
      ? undefined
      : {
          onDirectorChanged: (enabled) => hud.setDirector(enabled),
          onFocusChanged: (focus) => hud.setFocus(focus),
        },
  );

  const worker = new Worker(new URL('./physics/nbodyWorker.ts', import.meta.url), {
    type: 'module',
  });

  if (exportCfg) {
    hudRoot.style.display = 'none';
    void runOfflineExport({
      renderer,
      canvas,
      advect,
      cinemaRenderer,
      cinemaCamera,
      worker,
      config: exportCfg,
    });
    return;
  }

  function sendWorker(msg: WorkerInMessage): void {
    worker.postMessage(msg);
  }

  function requestStep(): void {
    if (!workerReady || workerInFlight || nbodyStepIndex >= MAX_NBODY_STEPS) return;
    workerInFlight = true;
    sendWorker({ type: 'step', id: ++stepReqId });
  }

  function requestRestore(slice: number): void {
    if (!workerReady || workerInFlight) {
      pendingRestoreSlice = slice;
      return;
    }
    pendingRestoreSlice = null;
    workerInFlight = true;
    sendWorker({ type: 'restoreCheckpoint', slot: slice });
  }

  function findBestCachedSlice(targetStep: number): number {
    let bestSlice = 0;
    let bestStep = 0;
    for (const [slice, sIdx] of cachedSlices) {
      if (sIdx <= targetStep && sIdx >= bestStep) {
        bestStep = sIdx;
        bestSlice = slice;
      }
    }
    return bestSlice;
  }

  function handleSeek(targetTimeGyr: number): void {
    const tClamped = Math.max(0.0, Math.min(T_MAX, targetTimeGyr));
    nbodyTimeDebt = 0;

    if (tClamped <= T_HANDOFF) {
      seekTargetStep = null;
      pendingRestoreSlice = null;
      simMode = 'approach';
      currentTimeGyr = tClamped;
      const st = getApproachState(tClamped);
      advect.setApproachState(tClamped, st.mwPos, st.mwVel, st.m31Pos, st.m31Vel);
      mwCenter = st.mwPos;
      m31Center = st.m31Pos;
      sepKpc = st.separationKpc;
      relSpeedKms = st.relSpeedKms;
      return;
    }

    const targetStep = Math.min(
      MAX_NBODY_STEPS,
      Math.max(0, Math.round((tClamped - T_HANDOFF) / DT)),
    );
    const bestSlice = findBestCachedSlice(targetStep);
    const bestSliceStep = cachedSlices.get(bestSlice) ?? 0;

    if (
      (simMode === 'nbody' || simMode === 'seeking' || simMode === 'boot_preroll') &&
      targetStep === nbodyStepIndex &&
      pendingRestoreSlice === null
    ) {
      seekTargetStep = null;
      simMode = 'nbody';
      return;
    }

    // If target is already directly ahead of our current N-body step with no closer checkpoint,
    // fast-forward directly without resetting backward!
    if (
      (simMode === 'nbody' || simMode === 'seeking') &&
      targetStep > nbodyStepIndex &&
      nbodyStepIndex >= bestSliceStep &&
      pendingRestoreSlice === null
    ) {
      seekTargetStep = targetStep;
      simMode = 'seeking';
      requestStep();
      return;
    }

    seekTargetStep = targetStep;
    simMode = 'seeking';
    requestRestore(bestSlice);
  }

  const hud = new MinimalHud(hudRoot, {
    onTogglePlay: () => {
      if (!playing && currentTimeGyr >= T_MAX - 1e-4) {
        handleSeek(0.0);
      }
      playing = !playing;
      hud.setPlaying(playing);
    },
    onCycleSpeed: () => {
      speedIdx = (speedIdx + 1) % SPEEDS.length;
      hud.setSpeedLabel(`${SPEEDS[speedIdx]}×`);
    },
    onSeekTime: handleSeek,
    onCycleFocus: () => {
      cinemaCamera.cycleFocus();
    },
    onToggleDirector: () => {
      cinemaCamera.toggleDirectorMode();
    },
    onResetCamera: () => {
      cinemaCamera.reset();
    },
  });

  worker.onmessage = (ev: MessageEvent<WorkerOutMessage>) => {
    const msg = ev.data;

    if (msg.type === 'inited') {
      workerReady = true;
      workerInFlight = false;
      advect.uploadOctree(msg.treePosMass, msg.treeMeta, msg.nodeCount);
      advect.resetToHandoff();
      advect.saveCheckpoint(0);
      cachedSlices.set(0, 0);
      maxCachedTimeGyr = T_HANDOFF;

      nbodyStepIndex = 0;
      currentTimeGyr = msg.timeGyr;
      gpuIsHalfShifted = false;
      mwCenter = msg.mwCenter;
      m31Center = msg.m31Center;
      sepKpc = msg.sepKpc;
      relSpeedKms = msg.relSpeedKms;

      // Kick off rapid boot pre-roll from T=2.60 -> T=2.96 Gyr
      if (seekTargetStep !== null && nbodyStepIndex < seekTargetStep) {
        requestStep();
      }
      return;
    }

    if (msg.type === 'restored') {
      workerInFlight = false;
      if (simMode === 'approach') return;
      advect.restoreCheckpoint(msg.slot);
      advect.uploadOctree(msg.treePosMass, msg.treeMeta, msg.nodeCount);
      nbodyStepIndex = msg.stepIndex;
      currentTimeGyr = msg.timeGyr;
      gpuIsHalfShifted = msg.stepIndex > 0;
      mwCenter = msg.mwCenter;
      m31Center = msg.m31Center;
      sepKpc = msg.sepKpc;
      relSpeedKms = msg.relSpeedKms;

      if (pendingRestoreSlice !== null) {
        requestRestore(pendingRestoreSlice);
        return;
      }

      if (seekTargetStep !== null && nbodyStepIndex < seekTargetStep) {
        requestStep();
      } else {
        seekTargetStep = null;
        simMode = 'nbody';
      }
      return;
    }

    if (msg.type === 'stepped') {
      workerInFlight = false;

      if (simMode === 'approach') return;

      // If user initiated a scrub restore while this step was in flight, discard and restore cleanly
      if (pendingRestoreSlice !== null) {
        requestRestore(pendingRestoreSlice);
        return;
      }

      // Exact Leapfrog GPU sync: step GPU using PREVIOUS uploaded octree T(x_n),
      // then upload NEW octree T(x_{n+1}) for the next step.
      const kickDt = gpuIsHalfShifted ? DT : 0.5 * DT;
      advect.step(kickDt, DT, DT, !gpuIsHalfShifted);
      gpuIsHalfShifted = true;
      advect.uploadOctree(msg.treePosMass, msg.treeMeta, msg.nodeCount);

      nbodyStepIndex = msg.stepIndex;
      currentTimeGyr = msg.timeGyr;
      mwCenter = msg.mwCenter;
      m31Center = msg.m31Center;
      sepKpc = msg.sepKpc;
      relSpeedKms = msg.relSpeedKms;

      if (msg.stepIndex % CHECKPOINT_STEP_STRIDE === 0) {
        const slice = Math.floor(msg.stepIndex / CHECKPOINT_STEP_STRIDE);
        if (slice < CHECKPOINT_SLICE_COUNT) {
          advect.saveCheckpoint(slice);
          sendWorker({ type: 'saveCheckpoint', slot: slice });
          cachedSlices.set(slice, msg.stepIndex);
          if (msg.timeGyr > maxCachedTimeGyr) {
            maxCachedTimeGyr = msg.timeGyr;
          }
        }
      }

      if (simMode === 'boot_preroll' || simMode === 'seeking') {
        if (seekTargetStep !== null && nbodyStepIndex < seekTargetStep) {
          requestStep();
        } else {
          if (simMode === 'boot_preroll') {
            hud.setBootReady();
            if (cinemaCamera.isDirectorMode()) {
              cinemaCamera.snapToDirector(
                currentTimeGyr,
                mwCenter,
                m31Center,
                sepKpc,
                performance.now() * 0.001,
              );
            }
          }
          seekTargetStep = null;
          simMode = 'nbody';
        }
      } else if (simMode === 'nbody') {
        nbodyTimeDebt = Math.max(0, nbodyTimeDebt - DT);
        if (currentTimeGyr >= T_MAX) {
          playing = false;
          hud.setPlaying(false);
          nbodyTimeDebt = 0;
        }
      }
    }
  };

  sendWorker({ type: 'init' });

  window.addEventListener('resize', () => {
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    cinemaCamera.setAspect(window.innerWidth / Math.max(1, window.innerHeight));
    cinemaRenderer.resize();
  });

  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
    const key = e.key.toLowerCase();
    if (e.code === 'Space') {
      e.preventDefault();
      if (!playing && currentTimeGyr >= T_MAX - 1e-4) {
        handleSeek(0.0);
      }
      playing = !playing;
      hud.setPlaying(playing);
    } else if (key === 'r') {
      cinemaCamera.reset();
    } else if (key === 'h') {
      hud.toggleHudVisibility();
    } else if (key === 'c' || key === 'd') {
      cinemaCamera.toggleDirectorMode();
    } else if (key === '1') {
      cinemaCamera.setFocus('COM');
    } else if (key === '2') {
      cinemaCamera.setFocus('MW');
    } else if (key === '3') {
      cinemaCamera.setFocus('M31');
    }
  });

  let lastTimeMs = performance.now();

  function animate(nowMs: number): void {
    requestAnimationFrame(animate);
    const dtSec = Math.min(0.05, Math.max(0.0001, (nowMs - lastTimeMs) / 1000.0));
    lastTimeMs = nowMs;

    if (playing) {
      const speed = SPEEDS[speedIdx];
      if (simMode === 'approach') {
        currentTimeGyr += dtSec * BASE_GYR_PER_SEC * speed;
        if (currentTimeGyr >= T_HANDOFF) {
          currentTimeGyr = T_HANDOFF;
          simMode = 'nbody';
          seekTargetStep = null;
          nbodyTimeDebt = 0;
          requestRestore(0);
        } else {
          const st = getApproachState(currentTimeGyr);
          advect.setApproachState(currentTimeGyr, st.mwPos, st.mwVel, st.m31Pos, st.m31Vel);
          mwCenter = st.mwPos;
          m31Center = st.m31Pos;
          sepKpc = st.separationKpc;
          relSpeedKms = st.relSpeedKms;
        }
      } else if (simMode === 'nbody' && currentTimeGyr < T_MAX) {
        nbodyTimeDebt = Math.min(DT * 3.0, nbodyTimeDebt + dtSec * BASE_GYR_PER_SEC * speed);
        if (nbodyTimeDebt >= DT && !workerInFlight) {
          requestStep();
        }
      }
    }

    const isSeeking = simMode === 'seeking' || simMode === 'boot_preroll';
    cinemaCamera.update(
      dtSec,
      currentTimeGyr,
      mwCenter,
      m31Center,
      sepKpc,
      nowMs * 0.001,
      isSeeking,
    );
    cinemaRenderer.render(cinemaCamera.camera, mwCenter, m31Center, sepKpc);
    hud.update(
      currentTimeGyr,
      Math.max(maxCachedTimeGyr, currentTimeGyr),
      sepKpc,
      relSpeedKms,
      isSeeking,
    );
  }

  requestAnimationFrame(animate);
}

bootstrap();
