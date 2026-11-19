import {
  BASIS_M31_VIEW,
  BASIS_MW_VIEW,
  DT,
  KPC_PER_GYR_TO_KMS,
  M31_BULGE_COUNT,
  M31_BULGE_START,
  M31_DISK_COUNT,
  M31_HALO_COUNT,
  M31_PARAMS,
  MW_BULGE_COUNT,
  MW_BULGE_START,
  MW_DISK_COUNT,
  MW_HALO_COUNT,
  MW_MACRO_COUNT,
  MW_PARAMS,
  N_MACRO,
  T_HANDOFF,
  Vec3,
} from './constants.ts';
import { getApproachState } from './approachOrbit.ts';
import { BarnesHutTree, findBulgeCenter, stepLeapfrogKDK } from './barnesHut.ts';
import { buildGalaxyRadialProfile, sampleMacroGalaxy } from './galaxyModel.ts';
import { createRng } from './random.ts';

export type WorkerInMessage =
  | { type: 'init' }
  | { type: 'step'; id: number }
  | { type: 'saveCheckpoint'; slot: number }
  | { type: 'restoreCheckpoint'; slot: number };

export interface SimulationSnapshot {
  timeGyr: number;
  stepIndex: number;
  nodeCount: number;
  treePosMass: Float32Array;
  treeMeta: Float32Array;
  mwCenter: Vec3;
  m31Center: Vec3;
  sepKpc: number;
  relSpeedKms: number;
}

export type WorkerOutMessage =
  | ({ type: 'inited' } & SimulationSnapshot)
  | ({ type: 'stepped'; id: number } & SimulationSnapshot)
  | ({ type: 'restored'; slot: number } & SimulationSnapshot);

interface MacroCheckpoint {
  pos: Float32Array;
  vel: Float32Array;
  stepIndex: number;
  timeGyr: number;
  mwCenter: Vec3;
  m31Center: Vec3;
  sepKpc: number;
  relSpeedKms: number;
}

const pos = new Float32Array(N_MACRO * 4);
const vel = new Float32Array(N_MACRO * 4);
const acc = new Float64Array(N_MACRO * 3);
const tree = new BarnesHutTree();
const checkpoints = new Map<number, MacroCheckpoint>();

let stepIndex = 0;
let timeGyr = T_HANDOFF;
let initialized = false;

function computeCoreVelocity(
  startIdx: number,
  count: number,
  center: Vec3,
): Vec3 {
  let vx = 0;
  let vy = 0;
  let vz = 0;
  let mSum = 0;
  const rCutSq = 16.0; // within 4 kpc of galactic core

  for (let i = 0; i < count; i++) {
    const b = (startIdx + i) * 4;
    const dx = pos[b] - center[0];
    const dy = pos[b + 1] - center[1];
    const dz = pos[b + 2] - center[2];
    if (dx * dx + dy * dy + dz * dz <= rCutSq) {
      const m = pos[b + 3];
      vx += m * vel[b];
      vy += m * vel[b + 1];
      vz += m * vel[b + 2];
      mSum += m;
    }
  }

  if (mSum <= 0) {
    for (let i = 0; i < count; i++) {
      const b = (startIdx + i) * 4;
      const m = pos[b + 3];
      vx += m * vel[b];
      vy += m * vel[b + 1];
      vz += m * vel[b + 2];
      mSum += m;
    }
  }

  return [vx / mSum, vy / mSum, vz / mSum];
}

function buildSnapshot(): {
  snapshot: SimulationSnapshot;
  transfers: Transferable[];
  mwCenter: Vec3;
  m31Center: Vec3;
  sepKpc: number;
  relSpeedKms: number;
} {
  const mwCenter = findBulgeCenter(pos, MW_BULGE_START, MW_BULGE_COUNT);
  const m31Center = findBulgeCenter(pos, M31_BULGE_START, M31_BULGE_COUNT);
  const sepKpc = Math.hypot(
    m31Center[0] - mwCenter[0],
    m31Center[1] - mwCenter[1],
    m31Center[2] - mwCenter[2],
  );

  const vMw = computeCoreVelocity(MW_BULGE_START, MW_BULGE_COUNT, mwCenter);
  const vM31 = computeCoreVelocity(M31_BULGE_START, M31_BULGE_COUNT, m31Center);
  const relSpeedKms =
    Math.hypot(vM31[0] - vMw[0], vM31[1] - vMw[1], vM31[2] - vMw[2]) *
    KPC_PER_GYR_TO_KMS;

  const floatLen = tree.nodeCount * 4;
  const treePosMass = tree.treePosMass.slice(0, floatLen);
  const treeMeta = tree.treeMeta.slice(0, floatLen);

  return {
    snapshot: {
      timeGyr,
      stepIndex,
      nodeCount: tree.nodeCount,
      treePosMass,
      treeMeta,
      mwCenter,
      m31Center,
      sepKpc,
      relSpeedKms,
    },
    transfers: [treePosMass.buffer, treeMeta.buffer],
    mwCenter,
    m31Center,
    sepKpc,
    relSpeedKms,
  };
}

function handleInit(): void {
  const mwProfile = buildGalaxyRadialProfile(MW_PARAMS);
  const m31Profile = buildGalaxyRadialProfile(M31_PARAMS);
  const handoff = getApproachState(T_HANDOFF);
  const rng = createRng(421337);

  const mwSample = sampleMacroGalaxy(
    mwProfile,
    MW_HALO_COUNT,
    MW_DISK_COUNT,
    MW_BULGE_COUNT,
    BASIS_MW_VIEW,
    handoff.mwPos,
    handoff.mwVel,
    rng,
  );
  const m31Sample = sampleMacroGalaxy(
    m31Profile,
    M31_HALO_COUNT,
    M31_DISK_COUNT,
    M31_BULGE_COUNT,
    BASIS_M31_VIEW,
    handoff.m31Pos,
    handoff.m31Vel,
    rng,
  );

  pos.set(mwSample.pos, 0);
  pos.set(m31Sample.pos, MW_MACRO_COUNT * 4);
  vel.set(mwSample.vel, 0);
  vel.set(m31Sample.vel, MW_MACRO_COUNT * 4);

  stepIndex = 0;
  timeGyr = T_HANDOFF;
  tree.build(pos, vel, N_MACRO);
  tree.computeAllAccels(pos, vel, acc, N_MACRO);
  initialized = true;

  const { snapshot, transfers, mwCenter, m31Center, sepKpc, relSpeedKms } =
    buildSnapshot();
  checkpoints.clear();
  checkpoints.set(0, {
    pos: new Float32Array(pos),
    vel: new Float32Array(vel),
    stepIndex: 0,
    timeGyr: T_HANDOFF,
    mwCenter,
    m31Center,
    sepKpc,
    relSpeedKms,
  });

  const msg: WorkerOutMessage = { type: 'inited', ...snapshot };
  self.postMessage(msg, transfers);
}

self.onmessage = (ev: MessageEvent<WorkerInMessage>): void => {
  const data = ev.data;
  if (data.type === 'init') {
    handleInit();
    return;
  }

  if (!initialized) return;

  if (data.type === 'step') {
    stepLeapfrogKDK(tree, pos, vel, acc, DT, N_MACRO, true);
    stepIndex++;
    timeGyr = T_HANDOFF + stepIndex * DT;
    const { snapshot, transfers } = buildSnapshot();
    const msg: WorkerOutMessage = { type: 'stepped', id: data.id, ...snapshot };
    self.postMessage(msg, transfers);
  } else if (data.type === 'saveCheckpoint') {
    const mwCenter = findBulgeCenter(pos, MW_BULGE_START, MW_BULGE_COUNT);
    const m31Center = findBulgeCenter(pos, M31_BULGE_START, M31_BULGE_COUNT);
    const sepKpc = Math.hypot(
      m31Center[0] - mwCenter[0],
      m31Center[1] - mwCenter[1],
      m31Center[2] - mwCenter[2],
    );
    const vMw = computeCoreVelocity(MW_BULGE_START, MW_BULGE_COUNT, mwCenter);
    const vM31 = computeCoreVelocity(M31_BULGE_START, M31_BULGE_COUNT, m31Center);
    const relSpeedKms =
      Math.hypot(vM31[0] - vMw[0], vM31[1] - vMw[1], vM31[2] - vMw[2]) *
      KPC_PER_GYR_TO_KMS;
    checkpoints.set(data.slot, {
      pos: new Float32Array(pos),
      vel: new Float32Array(vel),
      stepIndex,
      timeGyr,
      mwCenter,
      m31Center,
      sepKpc,
      relSpeedKms,
    });
  } else if (data.type === 'restoreCheckpoint') {
    const cp = checkpoints.get(data.slot);
    if (!cp) return;
    pos.set(cp.pos);
    vel.set(cp.vel);
    stepIndex = cp.stepIndex;
    timeGyr = cp.timeGyr;
    tree.build(pos, vel, N_MACRO);
    tree.computeAllAccels(pos, vel, acc, N_MACRO);
    const { snapshot, transfers } = buildSnapshot();
    const msg: WorkerOutMessage = {
      type: 'restored',
      slot: data.slot,
      ...snapshot,
    };
    self.postMessage(msg, transfers);
  }
};
