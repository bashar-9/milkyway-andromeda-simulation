import {
  BASIS_M31_VIEW,
  BASIS_MW_VIEW,
  DT,
  G,
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
  T_MAX,
} from './constants.ts';
import { getApproachState } from './approachOrbit.ts';
import { BarnesHutTree, findBulgeCenter, stepLeapfrogKDK } from './barnesHut.ts';
import { buildGalaxyRadialProfile, getCircularSpeedKms, sampleMacroGalaxy } from './galaxyModel.ts';
import { createRng } from './random.ts';

function assert(cond: boolean, msg: string): void {
  if (!cond) {
    throw new Error(`FAIL: ${msg}`);
  }
}

function computeTotalEnergyAndMomentum(
  pos: Float32Array,
  vel: Float32Array,
  count: number,
): { kinetic: number; potential: number; total: number; px: number; py: number; pz: number; scalarMom: number } {
  let ke = 0;
  let px = 0;
  let py = 0;
  let pz = 0;
  let scalarMom = 0;

  for (let i = 0; i < count; i++) {
    const b = i * 4;
    const m = pos[b + 3];
    const vx = vel[b];
    const vy = vel[b + 1];
    const vz = vel[b + 2];
    const v2 = vx * vx + vy * vy + vz * vz;
    ke += 0.5 * m * v2;
    px += m * vx;
    py += m * vy;
    pz += m * vz;
    scalarMom += m * Math.sqrt(v2);
  }

  // Exact direct O(N^2) pairwise gravitational potential energy
  let pe = 0;
  for (let i = 0; i < count; i++) {
    const bi = i * 4;
    const xi = pos[bi];
    const yi = pos[bi + 1];
    const zi = pos[bi + 2];
    const mi = pos[bi + 3];
    const epsi = vel[bi + 3];

    let sumPot = 0;
    for (let j = i + 1; j < count; j++) {
      const bj = j * 4;
      const dx = pos[bj] - xi;
      const dy = pos[bj + 1] - yi;
      const dz = pos[bj + 2] - zi;
      const epsPair = 0.5 * (epsi + vel[bj + 3]);
      const rSoft = Math.sqrt(dx * dx + dy * dy + dz * dz + epsPair);
      sumPot += pos[bj + 3] / rSoft;
    }
    pe -= G * mi * sumPot;
  }

  return {
    kinetic: ke,
    potential: pe,
    total: ke + pe,
    px,
    py,
    pz,
    scalarMom,
  };
}

function computeDiskHalfMassRadius(
  pos: Float32Array,
  startIdx: number,
  diskCount: number,
  center: [number, number, number],
): number {
  const radii = new Float64Array(diskCount);
  for (let i = 0; i < diskCount; i++) {
    const b = (startIdx + i) * 4;
    radii[i] = Math.hypot(
      pos[b] - center[0],
      pos[b + 1] - center[1],
      pos[b + 2] - center[2],
    );
  }
  radii.sort();
  return radii[Math.floor(diskCount * 0.5)];
}

console.log('=== 1. Galaxy Rotation Curve Verification ===');
const mwProfile = buildGalaxyRadialProfile(MW_PARAMS);
const m31Profile = buildGalaxyRadialProfile(M31_PARAMS);

const mwVc82 = getCircularSpeedKms(mwProfile, 8.2);
const m31Vc12 = getCircularSpeedKms(m31Profile, 12.0);
console.log(`MW  v_c(8.2 kpc)  = ${mwVc82.toFixed(2)} km/s (target: [215, 250])`);
console.log(`M31 v_c(12.0 kpc) = ${m31Vc12.toFixed(2)} km/s (target: [240, 280])`);
assert(mwVc82 >= 215 && mwVc82 <= 250, `MW v_c(8.2) out of range: ${mwVc82}`);
assert(m31Vc12 >= 240 && m31Vc12 <= 280, `M31 v_c(12) out of range: ${m31Vc12}`);

console.log('\n=== 2. Barnes-Hut Octree Accuracy vs Direct O(N^2) ===');
const handoff = getApproachState(T_HANDOFF);
console.log(
  `Approach T=${T_HANDOFF} Gyr: sep=${handoff.separationKpc.toFixed(2)} kpc, v_rel=${handoff.relSpeedKms.toFixed(2)} km/s`,
);

const rngInit = createRng(421337);
const mwSample = sampleMacroGalaxy(
  mwProfile,
  MW_HALO_COUNT,
  MW_DISK_COUNT,
  MW_BULGE_COUNT,
  BASIS_MW_VIEW,
  handoff.mwPos,
  handoff.mwVel,
  rngInit,
);
const m31Sample = sampleMacroGalaxy(
  m31Profile,
  M31_HALO_COUNT,
  M31_DISK_COUNT,
  M31_BULGE_COUNT,
  BASIS_M31_VIEW,
  handoff.m31Pos,
  handoff.m31Vel,
  rngInit,
);

const pos = new Float32Array(N_MACRO * 4);
const vel = new Float32Array(N_MACRO * 4);
const acc = new Float64Array(N_MACRO * 3);
pos.set(mwSample.pos, 0);
pos.set(m31Sample.pos, MW_MACRO_COUNT * 4);
vel.set(mwSample.vel, 0);
vel.set(m31Sample.vel, MW_MACRO_COUNT * 4);

const tree = new BarnesHutTree();
tree.build(pos, vel, N_MACRO);
console.log(`Flattened octree nodeCount = ${tree.nodeCount} (max allowed: 16384)`);
assert(tree.nodeCount <= 16384, `Octree nodeCount exceeded 16384: ${tree.nodeCount}`);

const testCount = 500;
const relErrors: number[] = [];
const tmpAcc = new Float64Array(3);
for (let k = 0; k < testCount; k++) {
  const i = Math.floor((k * N_MACRO) / testCount);
  const bi = i * 4;
  const xi = pos[bi];
  const yi = pos[bi + 1];
  const zi = pos[bi + 2];
  const epsi = vel[bi + 3];

  tree.computeAccelStackless(xi, yi, zi, epsi, tmpAcc, 0);

  let dax = 0;
  let day = 0;
  let daz = 0;
  for (let j = 0; j < N_MACRO; j++) {
    if (i === j) continue;
    const bj = j * 4;
    const dx = pos[bj] - xi;
    const dy = pos[bj + 1] - yi;
    const dz = pos[bj + 2] - zi;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 <= 1e-10) continue;
    const epsPair = 0.5 * (epsi + vel[bj + 3]);
    const denom = d2 + epsPair;
    const invR3 = pos[bj + 3] / (denom * Math.sqrt(denom));
    dax += dx * invR3;
    day += dy * invR3;
    daz += dz * invR3;
  }
  dax *= G;
  day *= G;
  daz *= G;

  const magDirect = Math.hypot(dax, day, daz);
  const err = Math.hypot(tmpAcc[0] - dax, tmpAcc[1] - day, tmpAcc[2] - daz);
  relErrors.push(err / Math.max(1e-8, magDirect));
}
relErrors.sort((a, b) => a - b);
const medianRelErr = relErrors[Math.floor(relErrors.length * 0.5)];
console.log(`Median relative force error across ${testCount} particles = ${(medianRelErr * 100).toFixed(3)}% (target: < 2.0%)`);
assert(medianRelErr < 0.02, `Octree force error too high: ${medianRelErr}`);

console.log('\n=== 3. Isolated Milky Way Disk Equilibrium & Energy Check (0.6 Gyr) ===');
const isoRng = createRng(9911);
const isoMw = sampleMacroGalaxy(
  mwProfile,
  MW_HALO_COUNT,
  MW_DISK_COUNT,
  MW_BULGE_COUNT,
  [1, 0, 0, 0, 1, 0, 0, 0, 1],
  [0, 0, 0],
  [0, 0, 0],
  isoRng,
);
const isoPos = new Float32Array(isoMw.pos);
const isoVel = new Float32Array(isoMw.vel);
const isoAcc = new Float64Array(MW_MACRO_COUNT * 3);
const isoTree = new BarnesHutTree();

const rHalf0 = computeDiskHalfMassRadius(isoPos, MW_HALO_COUNT, MW_DISK_COUNT, [0, 0, 0]);
const eIso0 = computeTotalEnergyAndMomentum(isoPos, isoVel, MW_MACRO_COUNT);

const isoSteps = Math.round(0.6 / DT);
for (let s = 0; s < isoSteps; s++) {
  stepLeapfrogKDK(isoTree, isoPos, isoVel, isoAcc, DT, MW_MACRO_COUNT, s > 0);
}
const cIso = findBulgeCenter(isoPos, MW_HALO_COUNT + MW_DISK_COUNT, MW_BULGE_COUNT);
const rHalf1 = computeDiskHalfMassRadius(isoPos, MW_HALO_COUNT, MW_DISK_COUNT, cIso);
const eIso1 = computeTotalEnergyAndMomentum(isoPos, isoVel, MW_MACRO_COUNT);
const isoDrift = Math.abs(rHalf1 - rHalf0) / rHalf0;
const isoEnergyErr = Math.abs(eIso1.total - eIso0.total) / Math.abs(eIso0.total);
console.log(`Isolated MW disk half-mass radius: ${rHalf0.toFixed(2)} -> ${rHalf1.toFixed(2)} kpc (drift: ${(isoDrift * 100).toFixed(2)}%, target <= 15%)`);
console.log(`Isolated MW energy drift |dE/E0| = ${(isoEnergyErr * 100).toFixed(3)}% (target < 1.5%)`);
assert(isoDrift <= 0.15, `Isolated MW disk drifted too much: ${isoDrift}`);
assert(isoEnergyErr < 0.015, `Isolated MW energy drift too large: ${isoEnergyErr}`);

console.log('\n=== 4. Full Milky Way x Andromeda Collision Simulation (T = 2.2 -> 7.5 Gyr) ===');
const eFull0 = computeTotalEnergyAndMomentum(pos, vel, N_MACRO);
const nSteps = Math.round((T_MAX - T_HANDOFF) / DT);
const sepHistory: { t: number; sep: number; vRelKms: number }[] = [];

const tStart = performance.now();
for (let s = 0; s <= nSteps; s++) {
  const t = T_HANDOFF + s * DT;
  const mwC = findBulgeCenter(pos, MW_BULGE_START, MW_BULGE_COUNT);
  const m31C = findBulgeCenter(pos, M31_BULGE_START, M31_BULGE_COUNT);
  const sep = Math.hypot(m31C[0] - mwC[0], m31C[1] - mwC[1], m31C[2] - mwC[2]);
  sepHistory.push({ t, sep, vRelKms: 0 });

  if (s < nSteps) {
    stepLeapfrogKDK(tree, pos, vel, acc, DT, N_MACRO, s > 0);
  }
}
const elapsedMs = performance.now() - tStart;
console.log(`Simulated ${nSteps} N-body steps in ${elapsedMs.toFixed(0)} ms (${(elapsedMs / nSteps).toFixed(2)} ms/step)`);

// Analyze orbital extrema (pericenter 1, apocenter 1, pericenter 2, coalescence)
let idxP1 = -1;
for (let i = 2; i < sepHistory.length - 2; i++) {
  if (
    sepHistory[i].sep < sepHistory[i - 1].sep &&
    sepHistory[i].sep <= sepHistory[i + 1].sep &&
    sepHistory[i].sep < 120
  ) {
    idxP1 = i;
    break;
  }
}
assert(idxP1 > 0, 'Did not detect First Pericenter!');
const p1 = sepHistory[idxP1];

let idxApo = idxP1 + 1;
for (let i = idxP1 + 1; i < sepHistory.length; i++) {
  if (sepHistory[i].sep > sepHistory[idxApo].sep) {
    idxApo = i;
  }
  // Stop searching apocenter once orbital decay plunges below 30 kpc
  if (i > idxP1 + 30 && sepHistory[i].sep < 30.0) break;
}
assert(idxApo > idxP1, 'Did not detect Apocenter after first pericenter!');
const apo = sepHistory[idxApo];

let idxP2 = -1;
for (let i = idxApo + 3; i < sepHistory.length - 2; i++) {
  if (
    sepHistory[i].sep < 25.0 &&
    sepHistory[i].sep < sepHistory[i - 1].sep &&
    sepHistory[i].sep <= sepHistory[i + 1].sep
  ) {
    idxP2 = i;
    break;
  }
}
const p2 = idxP2 > 0 ? sepHistory[idxP2] : null;

// Coalescence: first time separation remains below 10.5 kpc for remainder
let tCoal = sepHistory[sepHistory.length - 1].t;
for (let i = idxP1 + 10; i < sepHistory.length; i++) {
  let staysBelow = true;
  for (let j = i; j < sepHistory.length; j++) {
    if (sepHistory[j].sep > 10.5) {
      staysBelow = false;
      break;
    }
  }
  if (staysBelow) {
    tCoal = sepHistory[i].t;
    break;
  }
}

const finalSep = sepHistory[sepHistory.length - 1].sep;
const eFull1 = computeTotalEnergyAndMomentum(pos, vel, N_MACRO);
const fullEnergyErr = Math.abs(eFull1.total - eFull0.total) / Math.abs(eFull0.total);
const dP = Math.hypot(eFull1.px - eFull0.px, eFull1.py - eFull0.py, eFull1.pz - eFull0.pz);
const pDrift = dP / Math.max(1.0, eFull0.scalarMom);

console.log(`First Pericenter : T = ${p1.t.toFixed(3)} Gyr, sep = ${p1.sep.toFixed(2)} kpc (target T in [3.5,4.4], sep in [12,60])`);
console.log(`Apocenter        : T = ${apo.t.toFixed(3)} Gyr, sep = ${apo.sep.toFixed(2)} kpc (target > r_p1 + 30)`);
if (p2) {
  console.log(`Second Pericenter: T = ${p2.t.toFixed(3)} Gyr, sep = ${p2.sep.toFixed(2)} kpc`);
} else {
  console.log(`Second Pericenter: (direct rapid plunge into coalescence)`);
}
console.log(`Coalescence (<10.5 kpc): T = ${tCoal.toFixed(3)} Gyr`);
console.log(`Final Separation (T=${T_MAX} Gyr): ${finalSep.toFixed(2)} kpc (target < 12 kpc)`);
console.log(`Global Energy Drift |dE/E0| = ${(fullEnergyErr * 100).toFixed(3)}% (target < 2.5%)`);
console.log(`Global Momentum Drift = ${pDrift.toExponential(3)} (target < 1e-3)`);

// Also print separation at a few reference times
for (const checkT of [2.85, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]) {
  const idx = Math.min(sepHistory.length - 1, Math.round((checkT - T_HANDOFF) / DT));
  console.log(`  T=${sepHistory[idx].t.toFixed(2)} Gyr -> sep = ${sepHistory[idx].sep.toFixed(2)} kpc`);
}

assert(p1.t >= 3.5 && p1.t <= 4.4, `First pericenter time out of range: ${p1.t}`);
assert(p1.sep >= 12 && p1.sep <= 60, `First pericenter separation out of range: ${p1.sep}`);
assert(apo.sep > p1.sep + 30, `Apocenter not wide enough: ${apo.sep} vs ${p1.sep}`);
assert(finalSep < 12.0, `Galaxies did not coalesce by T=7.5 Gyr: finalSep=${finalSep}`);
assert(fullEnergyErr < 0.025, `Global energy drift too high: ${fullEnergyErr}`);
assert(pDrift < 1e-3, `Global momentum drift too high: ${pDrift}`);

console.log('\nALL PHYSICS VERIFICATION TESTS PASSED!');
