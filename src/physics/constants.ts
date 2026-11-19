// Natural galactic unit system:
//   Length = kpc
//   Mass   = 10^10 M_sun
//   Time   = Gyr
//
// Velocity conversion:
//   1 kpc / Gyr = 0.97779222 km/s
//   1 km/s      = 1.02271216 kpc / Gyr

export const G = 44985.0215; // kpc^3 / (10^10 M_sun * Gyr^2)
export const KPC_PER_GYR_TO_KMS = 0.97779222168;
export const KMS_TO_KPC_PER_GYR = 1.02271216503;

export interface GalaxyParams {
  readonly name: string;
  // Dark matter halo (truncated Hernquist)
  readonly mVirHalo: number;
  readonly aHalo: number;
  readonly rCutHalo: number;
  readonly fCutHalo: number;
  readonly mHaloInf: number;
  // Bulge (truncated Hernquist)
  readonly mBulge: number;
  readonly aBulge: number;
  readonly rCutBulge: number;
  readonly fCutBulge: number;
  readonly mBulgeInf: number;
  // Exponential disk
  readonly mDisk: number;
  readonly rDisk: number;
  readonly zDisk: number; // exponential z0; sech^2 scale height z_s = 2 * zDisk
  readonly rCutDisk: number;
  readonly fCutDisk: number;
  readonly mDiskInf: number;
  // Total physical truncated mass
  readonly mTotal: number;
  // Morphology metadata
  readonly barHalfLen: number;
  readonly barAngleRad: number;
  readonly pitchAngleDeg: number;
  readonly numMajorArms: number;
  readonly numMinorArms: number;
  readonly ringRadius: number;
  readonly ringWidth: number;
  // Softening lengths (kpc) for macro N-body particles
  readonly epsHalo: number;
  readonly epsDisk: number;
  readonly epsBulge: number;
}

function makeGalaxyParams(
  cfg: Omit<GalaxyParams, 'fCutHalo' | 'mHaloInf' | 'fCutBulge' | 'mBulgeInf' | 'fCutDisk' | 'mDiskInf' | 'mTotal'>,
): GalaxyParams {
  const fCutHalo = Math.pow(cfg.rCutHalo / (cfg.rCutHalo + cfg.aHalo), 2);
  const mHaloInf = cfg.mVirHalo / fCutHalo;

  const fCutBulge = Math.pow(cfg.rCutBulge / (cfg.rCutBulge + cfg.aBulge), 2);
  const mBulgeInf = cfg.mBulge / fCutBulge;

  const uCutDisk = cfg.rCutDisk / cfg.rDisk;
  const fCutDisk = 1 - (1 + uCutDisk) * Math.exp(-uCutDisk);
  const mDiskInf = cfg.mDisk / fCutDisk;

  const mTotal = cfg.mVirHalo + cfg.mBulge + cfg.mDisk;
  return {
    ...cfg,
    fCutHalo,
    mHaloInf,
    fCutBulge,
    mBulgeInf,
    fCutDisk,
    mDiskInf,
    mTotal,
  };
}

// Milky Way (MW) - calibrated to Gaia DR3 v_c(8.2 kpc) ~ 233 km/s, SBbc barred spiral
export const MW_PARAMS: GalaxyParams = makeGalaxyParams({
  name: 'Milky Way',
  mVirHalo: 105.0,
  aHalo: 42.0,
  rCutHalo: 210.0,
  mBulge: 1.6,
  aBulge: 0.8,
  rCutBulge: 12.0,
  mDisk: 5.4,
  rDisk: 3.0,
  zDisk: 0.35,
  rCutDisk: 22.0,
  barHalfLen: 4.0,
  barAngleRad: (28 * Math.PI) / 180,
  pitchAngleDeg: 13.0,
  numMajorArms: 2,
  numMinorArms: 2,
  ringRadius: 0.0,
  ringWidth: 0.0,
  epsHalo: 5.5,
  epsDisk: 1.2,
  epsBulge: 0.65,
});

// Andromeda (M31) - calibrated to peak v_c(12 kpc) ~ 259 km/s, SA(s)b + 10 kpc ring
export const M31_PARAMS: GalaxyParams = makeGalaxyParams({
  name: 'Andromeda (M31)',
  mVirHalo: 155.0,
  aHalo: 48.0,
  rCutHalo: 240.0,
  mBulge: 3.2,
  aBulge: 1.4,
  rCutBulge: 18.0,
  mDisk: 8.5,
  rDisk: 5.3,
  zDisk: 0.50,
  rCutDisk: 34.0,
  barHalfLen: 0.0,
  barAngleRad: 0.0,
  pitchAngleDeg: 9.5,
  numMajorArms: 2,
  numMinorArms: 2,
  ringRadius: 10.2,
  ringWidth: 1.35,
  epsHalo: 5.5,
  epsDisk: 1.2,
  epsBulge: 0.65,
});

// Real 3D HST/Gaia Initial Conditions at T = 0.0 Gyr (van der Marel et al. 2012, ApJ 753:8-9)
export const D0_KPC = 770.0;
export const V_RAD_KMS = -109.3;
export const V_TAN_KMS = 28.0; // Within 1-sigma observational bound (< 34.3 km/s)

export type Vec3 = [number, number, number];
export type Mat3 = [
  number, number, number,
  number, number, number,
  number, number, number,
];

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize3(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / len, v[1] / len, v[2] / len];
}

// Galactocentric unit vector MW -> M31 from (l = 121.17 deg, b = -21.57 deg)
const R_HAT_GAL: Vec3 = normalize3([-0.4814, 0.7957, -0.3676]);

// Tangential velocity direction in Galactocentric coords (van der Marel et al. 2012)
const T_RAW_GAL: Vec3 = [0.75, 0.60, 0.27];
const tDotR = dot3(T_RAW_GAL, R_HAT_GAL);
const T_HAT_GAL: Vec3 = normalize3([
  T_RAW_GAL[0] - tDotR * R_HAT_GAL[0],
  T_RAW_GAL[1] - tDotR * R_HAT_GAL[1],
  T_RAW_GAL[2] - tDotR * R_HAT_GAL[2],
]);
const N_ORB_GAL: Vec3 = normalize3(cross3(R_HAT_GAL, T_HAT_GAL));

// Build SO(3) rigid view transformation R_VIEW mapping Galactocentric -> Three.js Y-up view space:
//   R_HAT_GAL -> (1, 0, 0)
//   T_HAT_GAL -> (0, sin(10 deg), -cos(10 deg))
//   N_ORB_GAL -> (0, cos(10 deg),  sin(10 deg))
const TILT_RAD = (10.0 * Math.PI) / 180.0;
const sinT = Math.sin(TILT_RAD);
const cosT = Math.cos(TILT_RAD);

const V1: Vec3 = [1, 0, 0];
const V2: Vec3 = [0, sinT, -cosT];
const V3: Vec3 = [0, cosT, sinT];

// R_VIEW = V * E^T (row-major 3x3)
export const R_VIEW: Mat3 = [
  V1[0] * R_HAT_GAL[0] + V2[0] * T_HAT_GAL[0] + V3[0] * N_ORB_GAL[0],
  V1[0] * R_HAT_GAL[1] + V2[0] * T_HAT_GAL[1] + V3[0] * N_ORB_GAL[1],
  V1[0] * R_HAT_GAL[2] + V2[0] * T_HAT_GAL[2] + V3[0] * N_ORB_GAL[2],

  V1[1] * R_HAT_GAL[0] + V2[1] * T_HAT_GAL[0] + V3[1] * N_ORB_GAL[0],
  V1[1] * R_HAT_GAL[1] + V2[1] * T_HAT_GAL[1] + V3[1] * N_ORB_GAL[1],
  V1[1] * R_HAT_GAL[2] + V2[1] * T_HAT_GAL[2] + V3[1] * N_ORB_GAL[2],

  V1[2] * R_HAT_GAL[0] + V2[2] * T_HAT_GAL[0] + V3[2] * N_ORB_GAL[0],
  V1[2] * R_HAT_GAL[1] + V2[2] * T_HAT_GAL[1] + V3[2] * N_ORB_GAL[1],
  V1[2] * R_HAT_GAL[2] + V2[2] * T_HAT_GAL[2] + V3[2] * N_ORB_GAL[2],
];

export function applyMat3(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

// Construct orthonormal right-handed local disk basis [e1, e2, L] in View space
// where local (x, y) lies in disk plane and local +z is spin vector L.
export function buildDiskBasis(spinView: Vec3): Mat3 {
  const lz = normalize3(spinView);
  const ref: Vec3 = Math.abs(lz[0]) < 0.8 ? [1, 0, 0] : [0, 1, 0];
  const lx = normalize3(cross3(ref, lz));
  const ly = normalize3(cross3(lz, lx));
  // Columns are lx, ly, lz (row-major matrix)
  return [
    lx[0], ly[0], lz[0],
    lx[1], ly[1], lz[1],
    lx[2], ly[2], lz[2],
  ];
}

// Galaxy spin vectors in Galactocentric space -> rotated into View space
const L_MW_GAL: Vec3 = [0, 0, -1];
const L_M31_GAL: Vec3 = normalize3([0.431, -0.581, -0.690]);

export const L_MW_VIEW: Vec3 = normalize3(applyMat3(R_VIEW, L_MW_GAL));
export const L_M31_VIEW: Vec3 = normalize3(applyMat3(R_VIEW, L_M31_GAL));

export const BASIS_MW_VIEW: Mat3 = buildDiskBasis(L_MW_VIEW);
export const BASIS_M31_VIEW: Mat3 = buildDiskBasis(L_M31_VIEW);

// Relative separation and velocity at T = 0 in View space
export const R_HAT_VIEW: Vec3 = normalize3(applyMat3(R_VIEW, R_HAT_GAL));
export const T_HAT_VIEW: Vec3 = normalize3(applyMat3(R_VIEW, T_HAT_GAL));

export const REL_POS0_VIEW: Vec3 = [
  D0_KPC * R_HAT_VIEW[0],
  D0_KPC * R_HAT_VIEW[1],
  D0_KPC * R_HAT_VIEW[2],
];

export const REL_VEL0_VIEW: Vec3 = [
  (V_RAD_KMS * R_HAT_VIEW[0] + V_TAN_KMS * T_HAT_VIEW[0]) * KMS_TO_KPC_PER_GYR,
  (V_RAD_KMS * R_HAT_VIEW[1] + V_TAN_KMS * T_HAT_VIEW[1]) * KMS_TO_KPC_PER_GYR,
  (V_RAD_KMS * R_HAT_VIEW[2] + V_TAN_KMS * T_HAT_VIEW[2]) * KMS_TO_KPC_PER_GYR,
];

// Simulation timing & particle counts
export const T_HANDOFF = 2.60;       // Gyr: handoff from analytic RK4 approach to full N-body (~385 kpc sep)
export const T_MAX = 7.5;            // Gyr: full merger into Milkomeda
export const T_INITIAL_VIEW = 3.12;  // Gyr: initial view after quick boot pre-roll (~255 kpc sep)
export const DT = 0.004;             // Gyr (4.0 Myr per leapfrog step)

// Macro N-body counts (must sum to N_MACRO = 6144)
export const MW_HALO_COUNT = 1920;
export const MW_DISK_COUNT = 768;
export const MW_BULGE_COUNT = 384;
export const MW_MACRO_COUNT = MW_HALO_COUNT + MW_DISK_COUNT + MW_BULGE_COUNT; // 3072

export const M31_HALO_COUNT = 1920;
export const M31_DISK_COUNT = 768;
export const M31_BULGE_COUNT = 384;
export const M31_MACRO_COUNT = M31_HALO_COUNT + M31_DISK_COUNT + M31_BULGE_COUNT; // 3072

export const N_MACRO = MW_MACRO_COUNT + M31_MACRO_COUNT; // 6144

// Contiguous index spans for bulge core tracking
export const MW_BULGE_START = MW_HALO_COUNT + MW_DISK_COUNT;
export const M31_BULGE_START = MW_MACRO_COUNT + M31_HALO_COUNT + M31_DISK_COUNT;

// Barnes-Hut Octree & GPU buffers
export const THETA = 0.75;
export const THETA_SQ = THETA * THETA;
export const MAX_TREE_DEPTH = 14;
export const TREE_TEX_WIDTH = 128;
export const TREE_TEX_HEIGHT = 128;
export const MAX_TREE_NODES = TREE_TEX_WIDTH * TREE_TEX_HEIGHT; // 16384

// Visual GPU tracer textures (196,608 particles)
export const VIS_TEX_W = 512;
export const VIS_TEX_H = 384;
export const N_VISUAL = VIS_TEX_W * VIS_TEX_H; // 196,608

// Checkpoint VRAM array slices (covering T_HANDOFF..T_MAX)
export const CHECKPOINT_SLICE_COUNT = 24;
export const CHECKPOINT_STEP_STRIDE = 52; // ~208 Myr per checkpoint

export interface EpochMarker {
  readonly id: string;
  readonly label: string;
  readonly timeGyr: number;
}

// Calibrated directly from verifyPhysics.ts emergent N-body epochs
export const EPOCH_MARKERS: EpochMarker[] = [
  { id: 'today', label: 'Today', timeGyr: 0.0 },
  { id: 'impact1', label: '1st Impact', timeGyr: 3.82 },
  { id: 'tails', label: 'Tidal Tails', timeGyr: 4.43 },
  { id: 'impact2', label: '2nd Impact', timeGyr: 5.29 },
  { id: 'coalescence', label: 'Coalescence', timeGyr: 5.76 },
  { id: 'milkomeda', label: 'Milkomeda', timeGyr: 7.20 },
];
