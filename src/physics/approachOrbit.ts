import {
  G,
  KPC_PER_GYR_TO_KMS,
  M31_PARAMS,
  MW_PARAMS,
  REL_POS0_VIEW,
  REL_VEL0_VIEW,
  T_HANDOFF,
  Vec3,
} from './constants.ts';

export interface ApproachState {
  readonly timeGyr: number;
  readonly mwPos: Vec3;
  readonly mwVel: Vec3;
  readonly m31Pos: Vec3;
  readonly m31Vel: Vec3;
  readonly separationKpc: number;
  readonly relSpeedKms: number;
}

const DT_RK4 = 0.002; // 2 Myr RK4 step
const N_SAMPLES = Math.round(T_HANDOFF / DT_RK4) + 1; // 1101

const relPosTable = new Float64Array(N_SAMPLES * 3);
const relVelTable = new Float64Array(N_SAMPLES * 3);

// Mutual radial acceleration between MW and M31 during approach (T in [0, T_HANDOFF]).
// Beyond rCutSum = rCutHalo_MW + rCutHalo_M31 (450 kpc), spheres are strictly disjoint so
// Gauss's law gives exact point-mass force. Below 450 kpc, apply smooth outer-skirt overlap attenuation.
const R_CUT_SUM = MW_PARAMS.rCutHalo + M31_PARAMS.rCutHalo; // 450 kpc
const MU_SUM = G * (MW_PARAMS.mTotal + M31_PARAMS.mTotal);

function computeRelAccel(rx: number, ry: number, rz: number): Vec3 {
  const r2 = rx * rx + ry * ry + rz * rz;
  const r = Math.sqrt(r2);
  let overlapFactor = 1.0;
  if (r < R_CUT_SUM) {
    // Mild outer halo skirt overlap correction (smooth C^1 taper)
    const u = (R_CUT_SUM - r) / R_CUT_SUM;
    overlapFactor = 1.0 - 0.12 * u * u;
  }
  const factor = -(MU_SUM * overlapFactor) / (r2 * r);
  return [rx * factor, ry * factor, rz * factor];
}

// Integrate once at module init via 4th-order Runge-Kutta
(function buildApproachTable(): void {
  let rx = REL_POS0_VIEW[0];
  let ry = REL_POS0_VIEW[1];
  let rz = REL_POS0_VIEW[2];
  let vx = REL_VEL0_VIEW[0];
  let vy = REL_VEL0_VIEW[1];
  let vz = REL_VEL0_VIEW[2];

  for (let i = 0; i < N_SAMPLES; i++) {
    const base = i * 3;
    relPosTable[base] = rx;
    relPosTable[base + 1] = ry;
    relPosTable[base + 2] = rz;
    relVelTable[base] = vx;
    relVelTable[base + 1] = vy;
    relVelTable[base + 2] = vz;

    if (i === N_SAMPLES - 1) break;

    const h = DT_RK4;
    const hHalf = 0.5 * h;

    const [a1x, a1y, a1z] = computeRelAccel(rx, ry, rz);
    const k1rx = vx;
    const k1ry = vy;
    const k1rz = vz;

    const r2x = rx + hHalf * k1rx;
    const r2y = ry + hHalf * k1ry;
    const r2z = rz + hHalf * k1rz;
    const v2x = vx + hHalf * a1x;
    const v2y = vy + hHalf * a1y;
    const v2z = vz + hHalf * a1z;
    const [a2x, a2y, a2z] = computeRelAccel(r2x, r2y, r2z);

    const r3x = rx + hHalf * v2x;
    const r3y = ry + hHalf * v2y;
    const r3z = rz + hHalf * v2z;
    const v3x = vx + hHalf * a2x;
    const v3y = vy + hHalf * a2y;
    const v3z = vz + hHalf * a2z;
    const [a3x, a3y, a3z] = computeRelAccel(r3x, r3y, r3z);

    const r4x = rx + h * v3x;
    const r4y = ry + h * v3y;
    const r4z = rz + h * v3z;
    const v4x = vx + h * a3x;
    const v4y = vy + h * a3y;
    const v4z = vz + h * a3z;
    const [a4x, a4y, a4z] = computeRelAccel(r4x, r4y, r4z);

    rx += (h / 6.0) * (k1rx + 2.0 * v2x + 2.0 * v3x + v4x);
    ry += (h / 6.0) * (k1ry + 2.0 * v2y + 2.0 * v3y + v4y);
    rz += (h / 6.0) * (k1rz + 2.0 * v2z + 2.0 * v3z + v4z);

    vx += (h / 6.0) * (a1x + 2.0 * a2x + 2.0 * a3x + a4x);
    vy += (h / 6.0) * (a1y + 2.0 * a2y + 2.0 * a3y + a4y);
    vz += (h / 6.0) * (a1z + 2.0 * a2z + 2.0 * a3z + a4z);
  }
})();

const FRAC_MW = -M31_PARAMS.mTotal / (MW_PARAMS.mTotal + M31_PARAMS.mTotal);
const FRAC_M31 = MW_PARAMS.mTotal / (MW_PARAMS.mTotal + M31_PARAMS.mTotal);

// Cubic Hermite interpolation on the RK4 knot table
export function getApproachState(tGyr: number): ApproachState {
  const tClamped = Math.max(0.0, Math.min(T_HANDOFF, tGyr));
  const fIdx = tClamped / DT_RK4;
  const i0 = Math.min(N_SAMPLES - 2, Math.max(0, Math.floor(fIdx)));
  const i1 = i0 + 1;
  const s = fIdx - i0;

  const s2 = s * s;
  const s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + s;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;

  const b0 = i0 * 3;
  const b1 = i1 * 3;

  const p0x = relPosTable[b0];
  const p0y = relPosTable[b0 + 1];
  const p0z = relPosTable[b0 + 2];
  const p1x = relPosTable[b1];
  const p1y = relPosTable[b1 + 1];
  const p1z = relPosTable[b1 + 2];

  const v0x = relVelTable[b0];
  const v0y = relVelTable[b0 + 1];
  const v0z = relVelTable[b0 + 2];
  const v1x = relVelTable[b1];
  const v1y = relVelTable[b1 + 1];
  const v1z = relVelTable[b1 + 2];

  const rx = h00 * p0x + h10 * DT_RK4 * v0x + h01 * p1x + h11 * DT_RK4 * v1x;
  const ry = h00 * p0y + h10 * DT_RK4 * v0y + h01 * p1y + h11 * DT_RK4 * v1y;
  const rz = h00 * p0z + h10 * DT_RK4 * v0z + h01 * p1z + h11 * DT_RK4 * v1z;

  const [a0x, a0y, a0z] = computeRelAccel(p0x, p0y, p0z);
  const [a1x, a1y, a1z] = computeRelAccel(p1x, p1y, p1z);

  const vx = h00 * v0x + h10 * DT_RK4 * a0x + h01 * v1x + h11 * DT_RK4 * a1x;
  const vy = h00 * v0y + h10 * DT_RK4 * a0y + h01 * v1y + h11 * DT_RK4 * a1y;
  const vz = h00 * v0z + h10 * DT_RK4 * a0z + h01 * v1z + h11 * DT_RK4 * a1z;

  const mwPos: Vec3 = [FRAC_MW * rx, FRAC_MW * ry, FRAC_MW * rz];
  const mwVel: Vec3 = [FRAC_MW * vx, FRAC_MW * vy, FRAC_MW * vz];
  const m31Pos: Vec3 = [FRAC_M31 * rx, FRAC_M31 * ry, FRAC_M31 * rz];
  const m31Vel: Vec3 = [FRAC_M31 * vx, FRAC_M31 * vy, FRAC_M31 * vz];

  return {
    timeGyr: tClamped,
    mwPos,
    mwVel,
    m31Pos,
    m31Vel,
    separationKpc: Math.hypot(rx, ry, rz),
    relSpeedKms: Math.hypot(vx, vy, vz) * KPC_PER_GYR_TO_KMS,
  };
}
