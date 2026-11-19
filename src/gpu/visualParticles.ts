import {
  BASIS_M31_VIEW,
  BASIS_MW_VIEW,
  GalaxyParams,
  Mat3,
  N_VISUAL,
  Vec3,
  VIS_TEX_H,
  VIS_TEX_W,
  applyMat3,
} from '../physics/constants.ts';
import {
  GalaxyRadialProfile,
  evalDiskKinematics,
  evalProfileGrid,
  sampleExponentialRadius,
  sampleHernquistRadius,
  sampleSech2Z,
} from '../physics/galaxyModel.ts';
import { RNG, createRng, gaussian, randomUnitVector } from '../physics/random.ts';

export const VIS_SLICE_STARS_START = 0;
export const VIS_SLICE_STARS_COUNT = 140000;
export const VIS_SLICE_NEBULA_START = 140000;
export const VIS_SLICE_NEBULA_COUNT = 32000; // 24000 gas continuum + 8000 H-II nebulae
export const VIS_SLICE_DUST_START = 172000;
export const VIS_SLICE_DUST_COUNT = 24608;

export interface VisualParticlesData {
  readonly width: number;
  readonly height: number;
  readonly count: number;
  readonly initPos: Float32Array;      // (x, y, z, starburst=0)
  readonly initVel: Float32Array;      // (vx, vy, vz, baselineDensity=0)
  readonly staticAttr: Float32Array;   // (galId*10+popType, baseSizeKpc, luminosity, seedPhase)
  readonly staticColor: Float32Array;  // (r, g, b, localDiskOmega)
  readonly localInitPos: Float32Array; // (lx, ly, lz, galaxyId) for analytic approach scrub
}

const POP_OB_STAR = 0;
const POP_DISK_STAR = 1;
const POP_BULGE_STAR = 2;
const POP_HALO_GC = 3;
const POP_GAS_CONTINUUM = 4;
const POP_HII_NEBULA = 5;
const POP_DARK_DUST = 6;

interface GalaxyVisualSpec {
  galaxyId: 0 | 1;
  profile: GalaxyRadialProfile;
  basis: Mat3;
  centerPos: Vec3;
  centerVel: Vec3;
  nOB: number;
  nDisk: number;
  nBulge: number;
  nHaloGC: number;
  nGas: number;
  nHII: number;
  nDust: number;
}

function clampSpeed(
  vx: number,
  vy: number,
  vz: number,
  vEsc: number,
): [number, number, number] {
  const s = Math.hypot(vx, vy, vz);
  if (s > 0.94 * vEsc && s > 1e-8) {
    const k = (0.94 * vEsc) / s;
    return [vx * k, vy * k, vz * k];
  }
  return [vx, vy, vz];
}

function sampleArmAngle(
  R: number,
  p: GalaxyParams,
  armIndex: number,
  isMajor: boolean,
): number {
  const pitchRad = (p.pitchAngleDeg * Math.PI) / 180.0;
  const b = 1.0 / Math.tan(pitchRad);
  const rStart = p.barHalfLen > 0 ? p.barHalfLen * 0.82 : Math.max(1.2, p.aBulge * 1.4);
  const rClamped = Math.max(rStart, R);
  const baseOffset = p.barHalfLen > 0 ? p.barAngleRad : 0.35;
  const armBase = isMajor
    ? baseOffset + armIndex * Math.PI
    : baseOffset + 0.5 * Math.PI + armIndex * Math.PI + 0.25;
  // Trailing spiral: phi decreases as R increases
  return armBase - b * Math.log(rClamped / rStart);
}

function sampleGalaxyDiskPos(
  rng: RNG,
  p: GalaxyParams,
  mode: 'ob' | 'disk' | 'gas' | 'dust',
): { R: number; phi: number; z: number; isBar: boolean; isRing: boolean } {
  const hasBar = p.barHalfLen > 0;
  const hasRing = p.ringRadius > 0;

  // MW barred central region
  if (hasBar && ((mode === 'disk' && rng() < 0.22) || (mode === 'gas' && rng() < 0.18))) {
    const xb = (rng() * 2.0 - 1.0) * p.barHalfLen;
    const yb = gaussian(rng) * (0.75 + 0.15 * Math.abs(xb));
    const cosB = Math.cos(p.barAngleRad);
    const sinB = Math.sin(p.barAngleRad);
    const x = xb * cosB - yb * sinB;
    const y = xb * sinB + yb * cosB;
    const R = Math.max(0.25, Math.hypot(x, y));
    const phi = Math.atan2(y, x);
    const z = sampleSech2Z(rng, p.zDisk * 1.15);
    return { R, phi, z, isBar: true, isRing: false };
  }

  // M31 10 kpc star-forming ring
  if (
    hasRing &&
    ((mode === 'ob' && rng() < 0.36) ||
      (mode === 'gas' && rng() < 0.30) ||
      (mode === 'dust' && rng() < 0.34))
  ) {
    let R = p.ringRadius + gaussian(rng) * p.ringWidth;
    if (mode === 'dust') R -= 0.45; // inner edge of 10 kpc star-forming ring
    R = Math.max(2.0, Math.min(p.rCutDisk, R));
    const phi = 2.0 * Math.PI * rng();
    const zScale = mode === 'dust' ? 0.12 : mode === 'ob' ? p.zDisk * 0.45 : p.zDisk * 0.7;
    const z = sampleSech2Z(rng, zScale);
    return { R, phi, z, isBar: false, isRing: true };
  }

  // Sample radius from exponential profile (avoid tiny core for OB/dust)
  let R = sampleExponentialRadius(rng, p.rDisk, p.fCutDisk);
  if (mode === 'ob' || mode === 'dust') {
    let tries = 0;
    while (R < (hasBar ? 2.6 : 1.8) && tries < 6) {
      R = sampleExponentialRadius(rng, p.rDisk, p.fCutDisk);
      tries++;
    }
  }
  if (mode === 'ob') {
    R = Math.min(p.rCutDisk * 0.96, R * 1.08);
  }

  // Spiral density-wave placement
  let phi: number;
  const armWeight =
    mode === 'ob' ? 0.90 : mode === 'dust' ? 0.92 : mode === 'gas' ? 0.80 : 0.48;

  if (rng() < armWeight) {
    const isMajor = rng() < 0.72;
    const armIndex = rng() < 0.5 ? 0 : 1;
    const armPhi = sampleArmAngle(R, p, armIndex, isMajor);

    // Width of arm in radians (tighter for OB & dust, broader for old disk stars)
    const baseScatter =
      mode === 'ob'
        ? 0.16
        : mode === 'dust'
          ? 0.13
          : mode === 'gas'
            ? 0.24
            : 0.44;
    const radialWiden = 1.0 + 0.015 * R;
    // Dust sits slightly upstream (inner concave side) of spiral shock
    const shockOffset = mode === 'dust' ? -0.11 : mode === 'ob' ? 0.03 : 0.0;
    phi = armPhi + shockOffset + gaussian(rng) * baseScatter * radialWiden;
  } else {
    phi = 2.0 * Math.PI * rng();
  }

  const zHeight =
    mode === 'dust'
      ? 0.12
      : mode === 'ob'
        ? p.zDisk * 0.42
        : mode === 'gas'
          ? p.zDisk * 0.72
          : p.zDisk;
  const z = sampleSech2Z(rng, zHeight);

  return { R, phi, z, isBar: false, isRing: false };
}

function buildHIIKnots(
  rng: RNG,
  p: GalaxyParams,
  knotCount: number,
): Array<{ R: number; phi: number }> {
  const knots: Array<{ R: number; phi: number }> = [];
  for (let i = 0; i < knotCount; i++) {
    if (p.ringRadius > 0 && rng() < 0.42) {
      knots.push({
        R: Math.max(2.5, p.ringRadius + gaussian(rng) * (p.ringWidth * 0.8)),
        phi: 2.0 * Math.PI * rng(),
      });
    } else {
      let R = sampleExponentialRadius(rng, p.rDisk, p.fCutDisk);
      R = Math.max(p.barHalfLen > 0 ? 3.2 : 2.2, Math.min(p.rCutDisk * 0.9, R * 1.05));
      const isMajor = rng() < 0.76;
      const armIdx = rng() < 0.5 ? 0 : 1;
      const phi = sampleArmAngle(R, p, armIdx, isMajor) + 0.02 + gaussian(rng) * 0.10;
      knots.push({ R, phi });
    }
  }
  return knots;
}

export function generateVisualParticles(
  mwProfile: GalaxyRadialProfile,
  m31Profile: GalaxyRadialProfile,
  mwPos: Vec3,
  mwVel: Vec3,
  m31Pos: Vec3,
  m31Vel: Vec3,
): VisualParticlesData {
  const initPos = new Float32Array(N_VISUAL * 4);
  const initVel = new Float32Array(N_VISUAL * 4);
  const staticAttr = new Float32Array(N_VISUAL * 4);
  const staticColor = new Float32Array(N_VISUAL * 4);
  const localInitPos = new Float32Array(N_VISUAL * 4);

  const rng = createRng(7772026);

  const specs: GalaxyVisualSpec[] = [
    {
      galaxyId: 0,
      profile: mwProfile,
      basis: BASIS_MW_VIEW,
      centerPos: mwPos,
      centerVel: mwVel,
      nOB: 15500,
      nDisk: 29500,
      nBulge: 12860,
      nHaloGC: 140,
      nGas: 10000,
      nHII: 3400,
      nDust: 10608,
    },
    {
      galaxyId: 1,
      profile: m31Profile,
      basis: BASIS_M31_VIEW,
      centerPos: m31Pos,
      centerVel: m31Vel,
      nOB: 21500,
      nDisk: 41280,
      nBulge: 19000,
      nHaloGC: 220,
      nGas: 14000,
      nHII: 4600,
      nDust: 14000,
    },
  ];

  function writeParticle(
    slot: number,
    spec: GalaxyVisualSpec,
    lx: number,
    ly: number,
    lz: number,
    lvx: number,
    lvy: number,
    lvz: number,
    popType: number,
    sizeKpc: number,
    lum: number,
    r: number,
    g: number,
    b: number,
    omega: number,
  ): void {
    const pView = applyMat3(spec.basis, [lx, ly, lz]);
    const vView = applyMat3(spec.basis, [lvx, lvy, lvz]);
    const o = slot * 4;

    initPos[o] = pView[0] + spec.centerPos[0];
    initPos[o + 1] = pView[1] + spec.centerPos[1];
    initPos[o + 2] = pView[2] + spec.centerPos[2];
    initPos[o + 3] = 0.0; // starburst

    initVel[o] = vView[0] + spec.centerVel[0];
    initVel[o + 1] = vView[1] + spec.centerVel[1];
    initVel[o + 2] = vView[2] + spec.centerVel[2];
    initVel[o + 3] = 0.0; // baselineDensity (populated on step 0 on GPU)

    staticAttr[o] = spec.galaxyId * 10 + popType;
    staticAttr[o + 1] = sizeKpc;
    staticAttr[o + 2] = lum;
    staticAttr[o + 3] = rng();

    staticColor[o] = r;
    staticColor[o + 1] = g;
    staticColor[o + 2] = b;
    staticColor[o + 3] = omega;

    localInitPos[o] = lx;
    localInitPos[o + 1] = ly;
    localInitPos[o + 2] = lz;
    localInitPos[o + 3] = spec.galaxyId;
  }

  function emitDiskParticle(
    slot: number,
    spec: GalaxyVisualSpec,
    mode: 'ob' | 'disk' | 'gas' | 'dust',
    popType: number,
  ): void {
    const p = spec.profile.params;
    const { R, phi, z, isBar, isRing } = sampleGalaxyDiskPos(rng, p, mode);
    const cosP = Math.cos(phi);
    const sinP = Math.sin(phi);
    const lx = R * cosP;
    const ly = R * sinP;
    const lz = z;

    const kin = evalDiskKinematics(spec.profile, R);
    const dispScale = mode === 'ob' || mode === 'dust' ? 0.45 : mode === 'gas' ? 0.58 : 0.92;
    const vR = gaussian(rng) * kin.sigmaR * dispScale;
    const vPhi = kin.vPhiMean + gaussian(rng) * kin.sigmaPhi * dispScale;
    const vzRaw = gaussian(rng) * kin.sigmaZ * dispScale;
    const [vx, vy, vz] = clampSpeed(
      vR * cosP - vPhi * sinP,
      vR * sinP + vPhi * cosP,
      vzRaw,
      kin.vEsc,
    );

    if (mode === 'ob') {
      const tempMix = rng();
      const r = 0.24 + 0.22 * tempMix;
      const g = 0.54 + 0.22 * tempMix;
      const b = 1.0;
      const supergiant = rng() < 0.018;
      const sizeKpc = supergiant ? 0.45 + 0.30 * rng() : 0.16 + 0.22 * rng();
      const lum = supergiant ? 2.8 + 2.8 * rng() : 0.45 + 1.05 * Math.pow(rng(), 2.2);
      writeParticle(slot, spec, lx, ly, lz, vx, vy, vz, popType, sizeKpc, lum, r, g, b, kin.omega);
    } else if (mode === 'disk') {
      const rMix = Math.min(1.0, R / (2.5 * p.rDisk));
      const warm = rng() * 0.35 + (isBar ? 0.35 : 0.0) + 0.25 * (1.0 - rMix);
      const r = 1.0;
      const g = 0.90 - 0.16 * warm;
      const b = 0.76 - 0.32 * warm;
      const bright = rng() < 0.012;
      const sizeKpc = bright ? 0.34 + 0.22 * rng() : 0.13 + 0.17 * rng();
      const lum = bright ? 1.8 + 1.4 * rng() : 0.22 + 0.65 * Math.pow(rng(), 2.0);
      writeParticle(slot, spec, lx, ly, lz, vx, vy, vz, popType, sizeKpc, lum, r, g, b, kin.omega);
    } else if (mode === 'gas') {
      const innerMix = Math.exp(-R / (1.35 * p.rDisk));
      const blueMix = isRing ? 0.82 : Math.max(0.05, 1.0 - innerMix - (isBar ? 0.55 : 0.0));
      const r = 0.22 * blueMix + 1.0 * (1.0 - blueMix);
      const g = 0.52 * blueMix + 0.62 * (1.0 - blueMix);
      const b = 1.0 * blueMix + 0.26 * (1.0 - blueMix);
      const sizeKpc = 2.0 + 2.8 * rng();
      const lum = (0.35 + 0.55 * rng()) * (isRing ? 1.25 : 1.0);
      writeParticle(slot, spec, lx, ly, lz, vx, vy, vz, popType, sizeKpc, lum, r, g, b, kin.omega);
    } else {
      // Dark dust cloud
      const sizeKpc = 1.4 + 2.5 * rng();
      const opacityWeight = 0.45 + 0.55 * rng();
      writeParticle(slot, spec, lx, ly, lz, vx, vy, vz, popType, sizeKpc, opacityWeight, 1, 1, 1, kin.omega);
    }
  }

  function emitBulgeOrHaloParticle(
    slot: number,
    spec: GalaxyVisualSpec,
    kind: 'bulge' | 'haloGC' | 'bulgeGas',
  ): void {
    const p = spec.profile.params;
    const isHalo = kind === 'haloGC';
    const r = isHalo
      ? Math.max(8.0, Math.min(85.0, sampleHernquistRadius(rng, p.aHalo * 0.55, p.fCutHalo)))
      : sampleHernquistRadius(rng, p.aBulge, p.fCutBulge);
    const [nx, ny, nz] = randomUnitVector(rng);
    const lx = r * nx;
    const ly = r * ny;
    const lz = r * nz;

    const sig = evalProfileGrid(
      spec.profile,
      isHalo ? spec.profile.sigmaHaloGrid : spec.profile.sigmaBulgeGrid,
      r,
    );
    const vc = evalProfileGrid(spec.profile, spec.profile.vcSoftGrid, r);
    const vEsc = evalProfileGrid(spec.profile, spec.profile.vEscGrid, r);

    const Rxy = Math.hypot(lx, ly);
    const vRot = isHalo ? 0.05 * vc : 0.24 * vc;
    const rotVx = Rxy > 1e-5 ? (-vRot * ly) / Rxy : 0;
    const rotVy = Rxy > 1e-5 ? (vRot * lx) / Rxy : 0;

    const [vx, vy, vz] = clampSpeed(
      rotVx + gaussian(rng) * sig,
      rotVy + gaussian(rng) * sig,
      gaussian(rng) * sig,
      vEsc,
    );

    if (kind === 'bulge') {
      const m = rng();
      const cr = 1.0;
      const cg = 0.58 + 0.16 * m;
      const cb = 0.20 + 0.18 * m;
      const sizeKpc = 0.15 + 0.20 * rng();
      const lum = 0.32 + 0.85 * Math.pow(rng(), 2.1);
      writeParticle(slot, spec, lx, ly, lz, vx, vy, vz, POP_BULGE_STAR, sizeKpc, lum, cr, cg, cb, 0.0);
    } else if (kind === 'haloGC') {
      const cr = 1.0;
      const cg = 0.88 + 0.08 * rng();
      const cb = 0.68 + 0.14 * rng();
      const sizeKpc = 0.38 + 0.26 * rng();
      const lum = 1.3 + 1.5 * rng();
      writeParticle(slot, spec, lx, ly, lz, vx, vy, vz, POP_HALO_GC, sizeKpc, lum, cr, cg, cb, 0.0);
    } else {
      const cr = 1.0;
      const cg = 0.60 + 0.08 * rng();
      const cb = 0.24 + 0.10 * rng();
      const sizeKpc = 1.6 + 2.2 * rng();
      const lum = 0.40 + 0.50 * rng();
      writeParticle(slot, spec, lx, ly, lz, vx, vy, vz, POP_GAS_CONTINUUM, sizeKpc, lum, cr, cg, cb, 0.0);
    }
  }

  // -------------------------------------------------------------------------
  // Contiguous Partition 1: Crisp Stars [0 .. 139,999]
  // -------------------------------------------------------------------------
  let cursor = VIS_SLICE_STARS_START;
  for (const spec of specs) {
    for (let i = 0; i < spec.nOB; i++) emitDiskParticle(cursor++, spec, 'ob', POP_OB_STAR);
    for (let i = 0; i < spec.nDisk; i++) emitDiskParticle(cursor++, spec, 'disk', POP_DISK_STAR);
    for (let i = 0; i < spec.nBulge; i++) emitBulgeOrHaloParticle(cursor++, spec, 'bulge');
    for (let i = 0; i < spec.nHaloGC; i++) emitBulgeOrHaloParticle(cursor++, spec, 'haloGC');
  }
  if (cursor !== VIS_SLICE_NEBULA_START) {
    throw new Error(`Star partition mismatch: ${cursor} vs ${VIS_SLICE_NEBULA_START}`);
  }

  // -------------------------------------------------------------------------
  // Contiguous Partition 2: Volumetric Gas Continuum & H-II Nebulae [140,000 .. 171,999]
  // -------------------------------------------------------------------------
  for (const spec of specs) {
    const nBulgeGas = Math.floor(spec.nGas * 0.20);
    const nDiskGas = spec.nGas - nBulgeGas;
    for (let i = 0; i < nDiskGas; i++) emitDiskParticle(cursor++, spec, 'gas', POP_GAS_CONTINUUM);
    for (let i = 0; i < nBulgeGas; i++) emitBulgeOrHaloParticle(cursor++, spec, 'bulgeGas');
  }

  for (const spec of specs) {
    const p = spec.profile.params;
    const knots = buildHIIKnots(rng, p, 120);
    for (let i = 0; i < spec.nHII; i++) {
      const knot = knots[i % knots.length];
      const scatterKpc = 0.24 + 0.38 * rng();
      const a = 2.0 * Math.PI * rng();
      const xK = knot.R * Math.cos(knot.phi) + scatterKpc * Math.cos(a);
      const yK = knot.R * Math.sin(knot.phi) + scatterKpc * Math.sin(a);
      const R = Math.max(0.5, Math.hypot(xK, yK));
      const phi = Math.atan2(yK, xK);
      const cosP = Math.cos(phi);
      const sinP = Math.sin(phi);
      const z = sampleSech2Z(rng, p.zDisk * 0.35);

      const kin = evalDiskKinematics(spec.profile, R);
      const vR = gaussian(rng) * kin.sigmaR * 0.38;
      const vPhi = kin.vPhiMean + gaussian(rng) * kin.sigmaPhi * 0.38;
      const vzRaw = gaussian(rng) * kin.sigmaZ * 0.38;
      const [vx, vy, vz] = clampSpeed(
        vR * cosP - vPhi * sinP,
        vR * sinP + vPhi * cosP,
        vzRaw,
        kin.vEsc,
      );

      // 85% vibrant ruby-crimson H-alpha, 15% cyan-teal [O III] accents
      const isOIII = rng() < 0.15;
      const cr = isOIII ? 0.06 : 1.0;
      const cg = isOIII ? 0.78 : 0.08 + 0.08 * rng();
      const cb = isOIII ? 1.0 : 0.34 + 0.14 * rng();
      const sizeKpc = 0.50 + 0.95 * rng();
      const lum = 0.75 + 1.1 * rng();
      writeParticle(cursor++, spec, xK, yK, z, vx, vy, vz, POP_HII_NEBULA, sizeKpc, lum, cr, cg, cb, kin.omega);
    }
  }
  if (cursor !== VIS_SLICE_DUST_START) {
    throw new Error(`Nebula partition mismatch: ${cursor} vs ${VIS_SLICE_DUST_START}`);
  }

  // -------------------------------------------------------------------------
  // Contiguous Partition 3: Dark Molecular Dust Clouds [172,000 .. 196,607]
  // -------------------------------------------------------------------------
  for (const spec of specs) {
    for (let i = 0; i < spec.nDust; i++) {
      emitDiskParticle(cursor++, spec, 'dust', POP_DARK_DUST);
    }
  }
  if (cursor !== N_VISUAL) {
    throw new Error(`Total visual particle count mismatch: ${cursor} vs ${N_VISUAL}`);
  }

  return {
    width: VIS_TEX_W,
    height: VIS_TEX_H,
    count: N_VISUAL,
    initPos,
    initVel,
    staticAttr,
    staticColor,
    localInitPos,
  };
}
