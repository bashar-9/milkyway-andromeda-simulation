import {
  G,
  GalaxyParams,
  KPC_PER_GYR_TO_KMS,
  Mat3,
  Vec3,
  applyMat3,
} from './constants.ts';
import { RNG, gaussian, randomUnitVector } from './random.ts';

export interface GalaxyRadialProfile {
  readonly params: GalaxyParams;
  readonly nBins: number;
  readonly rGrid: Float64Array;
  readonly logRMin: number;
  readonly logRMax: number;
  readonly mTotGrid: Float64Array;
  readonly vcSoftGrid: Float64Array;     // analytic softened circular velocity (kpc/Gyr)
  readonly vcTrueGrid: Float64Array;     // physical unsoftened circular velocity (kpc/Gyr)
  readonly kappaGrid: Float64Array;      // epicyclic frequency (1/Gyr)
  readonly sigmaHaloGrid: Float64Array;  // isotropic Jeans velocity dispersion for halo (kpc/Gyr)
  readonly sigmaBulgeGrid: Float64Array; // isotropic Jeans velocity dispersion for bulge (kpc/Gyr)
  readonly vEscGrid: Float64Array;       // escape speed (kpc/Gyr)
}

// Modified Bessel functions I0, I1, K0, K1 (Abramowitz & Stegun 9.8, ~1e-7 accuracy)
function besselI0(x: number): number {
  const ax = Math.abs(x);
  if (ax < 3.75) {
    const t = ax / 3.75;
    const t2 = t * t;
    return (
      1.0 +
      t2 *
        (3.5156229 +
          t2 *
            (3.0899424 +
              t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))))
    );
  }
  const t = 3.75 / ax;
  return (
    (Math.exp(ax) / Math.sqrt(ax)) *
    (0.39894228 +
      t *
        (0.01328592 +
          t *
            (0.00225319 +
              t *
                (-0.00157565 +
                  t *
                    (0.00916281 +
                      t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
  );
}

function besselI1(x: number): number {
  const ax = Math.abs(x);
  let ans: number;
  if (ax < 3.75) {
    const t = ax / 3.75;
    const t2 = t * t;
    ans =
      ax *
      (0.5 +
        t2 *
          (0.87890594 +
            t2 *
              (0.51498869 +
                t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))));
  } else {
    const t = 3.75 / ax;
    ans =
      (Math.exp(ax) / Math.sqrt(ax)) *
      (0.39894228 +
        t *
          (-0.03988024 +
            t *
              (-0.00362018 +
                t *
                  (0.00163801 +
                    t *
                      (-0.01031555 +
                        t * (0.02282967 + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059))))))));
  }
  return x < 0 ? -ans : ans;
}

function besselK0(x: number): number {
  if (x <= 2.0) {
    const y = (x * x) / 4.0;
    return (
      -Math.log(x / 2.0) * besselI0(x) +
      (-0.57721566 +
        y *
          (0.4227842 +
            y *
              (0.23069756 +
                y * (0.0348859 + y * (0.00262698 + y * (0.0001075 + y * 0.0000074))))))
    );
  }
  const y = 2.0 / x;
  return (
    (Math.exp(-x) / Math.sqrt(x)) *
    (1.25331414 +
      y *
        (-0.07832358 +
          y *
            (0.02189568 +
              y * (-0.01062446 + y * (0.00587872 + y * (-0.0025154 + y * 0.00053208))))))
  );
}

function besselK1(x: number): number {
  if (x <= 2.0) {
    const y = (x * x) / 4.0;
    return (
      Math.log(x / 2.0) * besselI1(x) +
      (1.0 / x) *
        (1.0 +
          y *
            (0.15443144 +
              y *
                (-0.67278579 +
                  y *
                    (-0.18156897 +
                      y * (-0.01919402 + y * (-0.00110404 - y * 0.00004686))))))
    );
  }
  const y = 2.0 / x;
  return (
    (Math.exp(-x) / Math.sqrt(x)) *
    (1.25331414 +
      y *
        (0.23498619 +
          y *
            (-0.0365562 +
              y * (0.01504268 + y * (-0.00780353 + y * (0.00325614 - y * 0.00068245))))))
  );
}

export function hernquistMassEnclosed(r: number, mInf: number, a: number, rCut: number, mTrunc: number): number {
  if (r <= 0) return 0;
  if (r >= rCut) return mTrunc;
  const f = r / (r + a);
  return mInf * f * f;
}

export function hernquistDensity(r: number, mInf: number, a: number, rCut: number): number {
  if (r <= 0 || r >= rCut) return 0;
  const ra = r + a;
  return (mInf * a) / (2.0 * Math.PI * r * ra * ra * ra);
}

export function expDiskMassEnclosed(r: number, mDiskInf: number, rDisk: number, rCut: number, mTrunc: number): number {
  if (r <= 0) return 0;
  if (r >= rCut) return mTrunc;
  const u = r / rDisk;
  return mDiskInf * (1.0 - (1.0 + u) * Math.exp(-u));
}

function freemanVcSq(R: number, mDisk: number, rDisk: number): number {
  if (R <= 1e-4) return 0;
  const y = R / (2.0 * rDisk);
  if (y > 22.0) {
    return (G * mDisk) / R;
  }
  const term = besselI0(y) * besselK0(y) - besselI1(y) * besselK1(y);
  return ((2.0 * G * mDisk) / rDisk) * y * y * Math.max(0, term);
}

export function buildGalaxyRadialProfile(params: GalaxyParams): GalaxyRadialProfile {
  const nBins = 256;
  const rMin = 0.04;
  const rMax = params.rCutHalo;
  const logRMin = Math.log(rMin);
  const logRMax = Math.log(rMax);

  const rGrid = new Float64Array(nBins);
  const mTotGrid = new Float64Array(nBins);
  const vcSoftGrid = new Float64Array(nBins);
  const vcTrueGrid = new Float64Array(nBins);
  const kappaGrid = new Float64Array(nBins);
  const sigmaHaloGrid = new Float64Array(nBins);
  const sigmaBulgeGrid = new Float64Array(nBins);
  const vEscGrid = new Float64Array(nBins);

  const epsDiskHaloSq = 0.5 * (params.epsDisk * params.epsDisk + params.epsHalo * params.epsHalo);
  const epsDiskBulgeSq = 0.5 * (params.epsDisk * params.epsDisk + params.epsBulge * params.epsBulge);
  const zScale = 2.0 * params.zDisk;
  const epsDiskSelfSq = params.epsDisk * params.epsDisk + (0.65 * zScale) * (0.65 * zScale);

  for (let i = 0; i < nBins; i++) {
    const t = i / (nBins - 1);
    const r = Math.exp(logRMin + t * (logRMax - logRMin));
    rGrid[i] = r;

    const mH = hernquistMassEnclosed(r, params.mHaloInf, params.aHalo, params.rCutHalo, params.mVirHalo);
    const mB = hernquistMassEnclosed(r, params.mBulgeInf, params.aBulge, params.rCutBulge, params.mBulge);
    const mD = expDiskMassEnclosed(r, params.mDiskInf, params.rDisk, params.rCutDisk, params.mDisk);
    mTotGrid[i] = mH + mB + mD;

    const vcSqTrueHalo = (G * mH) / r;
    const vcSqTrueBulge = (G * mB) / r;
    const vcSqTrueDisk =
      r <= params.rCutDisk
        ? freemanVcSq(r, params.mDiskInf, params.rDisk)
        : (G * params.mDisk) / r;
    vcTrueGrid[i] = Math.sqrt(Math.max(0, vcSqTrueHalo + vcSqTrueBulge + vcSqTrueDisk));

    const rEffD = Math.sqrt(r * r + epsDiskSelfSq);
    const vcSqSoftHalo = (G * mH * r * r) / Math.pow(r * r + epsDiskHaloSq, 1.5);
    const vcSqSoftBulge = (G * mB * r * r) / Math.pow(r * r + epsDiskBulgeSq, 1.5);
    const vcSqSoftDisk = vcSqTrueDisk * Math.pow(r / rEffD, 3.0);
    vcSoftGrid[i] = Math.sqrt(Math.max(0, vcSqSoftHalo + vcSqSoftBulge + vcSqSoftDisk));
  }

  for (let i = 0; i < nBins; i++) {
    const r = rGrid[i];
    const vc = vcSoftGrid[i];
    const omega = vc / r;
    let dVcDr: number;
    if (i === 0) {
      dVcDr = (vcSoftGrid[1] - vcSoftGrid[0]) / (rGrid[1] - rGrid[0]);
    } else if (i === nBins - 1) {
      dVcDr = (vcSoftGrid[nBins - 1] - vcSoftGrid[nBins - 2]) / (rGrid[nBins - 1] - rGrid[nBins - 2]);
    } else {
      dVcDr = (vcSoftGrid[i + 1] - vcSoftGrid[i - 1]) / (rGrid[i + 1] - rGrid[i - 1]);
    }
    const kappaSq = Math.max(0.25 * omega * omega, 2.0 * omega * (omega + dVcDr));
    kappaGrid[i] = Math.sqrt(kappaSq);
  }

  let intHalo = 0.0;
  let intBulge = 0.0;
  let phi = -(G * params.mTotal) / rMax;
  const epsH2 = params.epsHalo * params.epsHalo;
  const epsB2 = params.epsBulge * params.epsBulge;

  vEscGrid[nBins - 1] = Math.sqrt(Math.max(0, -2.0 * phi));
  sigmaHaloGrid[nBins - 1] = 0;
  sigmaBulgeGrid[nBins - 1] = 0;

  for (let i = nBins - 2; i >= 0; i--) {
    const r0 = rGrid[i];
    const r1 = rGrid[i + 1];
    const dr = r1 - r0;

    const fPhi0 = (G * mTotGrid[i]) / (r0 * r0 + 1.0);
    const fPhi1 = (G * mTotGrid[i + 1]) / (r1 * r1 + 1.0);
    phi -= 0.5 * (fPhi0 + fPhi1) * dr;
    vEscGrid[i] = Math.sqrt(Math.max(0, -2.0 * phi));

    const rhoH0 = hernquistDensity(r0, params.mHaloInf, params.aHalo, params.rCutHalo);
    const rhoH1 = hernquistDensity(r1, params.mHaloInf, params.aHalo, params.rCutHalo);
    const jH0 = rhoH0 * ((G * mTotGrid[i]) / (r0 * r0 + epsH2));
    const jH1 = rhoH1 * ((G * mTotGrid[i + 1]) / (r1 * r1 + epsH2));
    intHalo += 0.5 * (jH0 + jH1) * dr;
    sigmaHaloGrid[i] = rhoH0 > 1e-18 ? Math.sqrt(Math.max(0, intHalo / rhoH0)) : 0;

    if (r0 < params.rCutBulge) {
      const rhoB0 = hernquistDensity(r0, params.mBulgeInf, params.aBulge, params.rCutBulge);
      const rhoB1 = hernquistDensity(r1, params.mBulgeInf, params.aBulge, params.rCutBulge);
      const jB0 = rhoB0 * ((G * mTotGrid[i]) / (r0 * r0 + epsB2));
      const jB1 = rhoB1 * ((G * mTotGrid[i + 1]) / (r1 * r1 + epsB2));
      intBulge += 0.5 * (jB0 + jB1) * dr;
      sigmaBulgeGrid[i] = rhoB0 > 1e-18 ? Math.sqrt(Math.max(0, intBulge / rhoB0)) : 0;
    } else {
      sigmaBulgeGrid[i] = 0;
    }
  }

  return {
    params,
    nBins,
    rGrid,
    logRMin,
    logRMax,
    mTotGrid,
    vcSoftGrid,
    vcTrueGrid,
    kappaGrid,
    sigmaHaloGrid,
    sigmaBulgeGrid,
    vEscGrid,
  };
}

export function evalProfileGrid(profile: GalaxyRadialProfile, grid: Float64Array, r: number): number {
  const logR = Math.log(Math.max(profile.rGrid[0], Math.min(profile.rGrid[profile.nBins - 1], r)));
  const u = ((logR - profile.logRMin) / (profile.logRMax - profile.logRMin)) * (profile.nBins - 1);
  const idx = Math.min(profile.nBins - 2, Math.max(0, Math.floor(u)));
  const frac = u - idx;
  return grid[idx] * (1.0 - frac) + grid[idx + 1] * frac;
}

export function sampleHernquistRadius(rng: RNG, a: number, fCut: number): number {
  const u = Math.max(1e-8, Math.min(0.999999, rng())) * fCut;
  const s = Math.sqrt(u);
  return (a * s) / Math.max(1e-6, 1.0 - s);
}

export function sampleExponentialRadius(rng: RNG, rDisk: number, fCut: number): number {
  const target = Math.max(1e-7, Math.min(0.999999, rng())) * fCut;
  let x = target < 0.3 ? Math.sqrt(2.0 * target) : -Math.log(Math.max(1e-6, 1.0 - target));
  for (let iter = 0; iter < 10; iter++) {
    const ex = Math.exp(-x);
    const f = 1.0 - (1.0 + x) * ex - target;
    const fp = x * ex;
    if (fp < 1e-12) break;
    x = Math.max(1e-5, x - f / fp);
  }
  return x * rDisk;
}

export function sampleSech2Z(rng: RNG, zDisk: number): number {
  const zScale = 2.0 * zDisk;
  const u = 1e-4 + rng() * (1.0 - 2e-4);
  return 0.5 * zScale * Math.log(u / (1.0 - u));
}

export interface DiskKinematics {
  readonly omega: number;
  readonly vPhiMean: number;
  readonly sigmaR: number;
  readonly sigmaPhi: number;
  readonly sigmaZ: number;
  readonly vEsc: number;
}

// Compute exact equilibrium profile calibrated directly against the sampled discrete
// Plummer-softened N-body force field so halo, disk, and bulge have zero breathing modes.
export function buildSoftenedDiscreteProfile(
  analytic: GalaxyRadialProfile,
  pos: Float32Array,
  vel: Float32Array,
  total: number,
): GalaxyRadialProfile {
  const p = analytic.params;
  const nBins = analytic.nBins;
  const rGrid = analytic.rGrid;

  const nKnots = 48;
  const kR = new Float64Array(nKnots);
  const kLogR = new Float64Array(nKnots);
  const kVcDisk = new Float64Array(nKnots);
  const kForceHalo = new Float64Array(nKnots);
  const kForceBulge = new Float64Array(nKnots);

  const epsH2 = p.epsHalo * p.epsHalo;
  const epsD2 = p.epsDisk * p.epsDisk;
  const epsB2 = p.epsBulge * p.epsBulge;

  // 8 azimuthal directions in midplane for disk; 8 spherical directions for halo/bulge
  const fibDirs: [number, number, number][] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [0.57735, 0.57735, 0.57735],
    [-0.57735, -0.57735, -0.57735],
  ];

  for (let k = 0; k < nKnots; k++) {
    const r = rGrid[0] * Math.pow(rGrid[nBins - 1] / rGrid[0], k / (nKnots - 1));
    kR[k] = r;
    kLogR[k] = Math.log(r);

    // Midplane disk radial force (8 azimuths)
    let arDisk = 0;
    for (let phiIdx = 0; phiIdx < 8; phiIdx++) {
      const ang = (phiIdx * Math.PI) / 4.0;
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);
      const x = r * ca;
      const y = r * sa;
      let sum = 0;
      for (let j = 0; j < total; j++) {
        const jb = j * 4;
        const dx = pos[jb] - x;
        const dy = pos[jb + 1] - y;
        const dz = pos[jb + 2];
        const d2 = dx * dx + dy * dy + dz * dz;
        const ep = 0.5 * (epsD2 + vel[jb + 3]);
        const denom = d2 + ep;
        const inv = pos[jb + 3] / (denom * Math.sqrt(denom));
        sum -= (dx * ca + dy * sa) * inv;
      }
      arDisk += sum;
    }
    kVcDisk[k] = Math.sqrt(Math.max(0, (G * arDisk * r) / 8.0));

    // Spherical radial force for halo and bulge particles (8 directions)
    let frHalo = 0;
    let frBulge = 0;
    for (let dIdx = 0; dIdx < 8; dIdx++) {
      const [nx, ny, nz] = fibDirs[dIdx];
      const x = r * nx;
      const y = r * ny;
      const z = r * nz;
      let sH = 0;
      let sB = 0;
      for (let j = 0; j < total; j++) {
        const jb = j * 4;
        const dx = pos[jb] - x;
        const dy = pos[jb + 1] - y;
        const dz = pos[jb + 2] - z;
        const d2 = dx * dx + dy * dy + dz * dz;
        const dotN = dx * nx + dy * ny + dz * nz;
        const m = pos[jb + 3];
        const ej2 = vel[jb + 3];

        const dH = d2 + 0.5 * (epsH2 + ej2);
        sH -= (m * dotN) / (dH * Math.sqrt(dH));

        const dB = d2 + 0.5 * (epsB2 + ej2);
        sB -= (m * dotN) / (dB * Math.sqrt(dB));
      }
      frHalo += sH;
      frBulge += sB;
    }
    kForceHalo[k] = Math.max(0, (G * frHalo) / 8.0);
    kForceBulge[k] = Math.max(0, (G * frBulge) / 8.0);
  }

  function interpKnot(arr: Float64Array, r: number): number {
    const lr = Math.log(Math.max(kR[0], Math.min(kR[nKnots - 1], r)));
    const u = ((lr - kLogR[0]) / (kLogR[nKnots - 1] - kLogR[0])) * (nKnots - 1);
    const idx = Math.min(nKnots - 2, Math.max(0, Math.floor(u)));
    const f = u - idx;
    return arr[idx] * (1.0 - f) + arr[idx + 1] * f;
  }

  const vcSoftGrid = new Float64Array(nBins);
  const kappaGrid = new Float64Array(nBins);
  const sigmaHaloGrid = new Float64Array(nBins);
  const sigmaBulgeGrid = new Float64Array(nBins);
  const vEscGrid = new Float64Array(nBins);
  const fHaloGrid = new Float64Array(nBins);
  const fBulgeGrid = new Float64Array(nBins);

  for (let i = 0; i < nBins; i++) {
    const r = rGrid[i];
    vcSoftGrid[i] = interpKnot(kVcDisk, r);
    fHaloGrid[i] = interpKnot(kForceHalo, r);
    fBulgeGrid[i] = interpKnot(kForceBulge, r);
  }

  for (let i = 0; i < nBins; i++) {
    const r = rGrid[i];
    const vc = vcSoftGrid[i];
    const omega = vc / r;
    let dVcDr: number;
    if (i === 0) {
      dVcDr = (vcSoftGrid[1] - vcSoftGrid[0]) / (rGrid[1] - rGrid[0]);
    } else if (i === nBins - 1) {
      dVcDr = (vcSoftGrid[nBins - 1] - vcSoftGrid[nBins - 2]) / (rGrid[nBins - 1] - rGrid[nBins - 2]);
    } else {
      dVcDr = (vcSoftGrid[i + 1] - vcSoftGrid[i - 1]) / (rGrid[i + 1] - rGrid[i - 1]);
    }
    const kappaSq = Math.max(0.25 * omega * omega, 2.0 * omega * (omega + dVcDr));
    kappaGrid[i] = Math.sqrt(kappaSq);
  }

  let intHalo = 0.0;
  let intBulge = 0.0;
  let phi = -(G * p.mTotal) / rGrid[nBins - 1];
  vEscGrid[nBins - 1] = Math.sqrt(Math.max(0, -2.0 * phi));

  for (let i = nBins - 2; i >= 0; i--) {
    const r0 = rGrid[i];
    const r1 = rGrid[i + 1];
    const dr = r1 - r0;

    phi -= 0.5 * (fBulgeGrid[i] + fBulgeGrid[i + 1]) * dr;
    vEscGrid[i] = Math.sqrt(Math.max(0, -2.0 * phi));

    const rhoH0 = hernquistDensity(r0, p.mHaloInf, p.aHalo, p.rCutHalo);
    const rhoH1 = hernquistDensity(r1, p.mHaloInf, p.aHalo, p.rCutHalo);
    intHalo += 0.5 * (rhoH0 * fHaloGrid[i] + rhoH1 * fHaloGrid[i + 1]) * dr;
    sigmaHaloGrid[i] = rhoH0 > 1e-18 ? Math.sqrt(Math.max(0, intHalo / rhoH0)) : 0;

    if (r0 < p.rCutBulge) {
      const rhoB0 = hernquistDensity(r0, p.mBulgeInf, p.aBulge, p.rCutBulge);
      const rhoB1 = hernquistDensity(r1, p.mBulgeInf, p.aBulge, p.rCutBulge);
      intBulge += 0.5 * (rhoB0 * fBulgeGrid[i] + rhoB1 * fBulgeGrid[i + 1]) * dr;
      sigmaBulgeGrid[i] = rhoB0 > 1e-18 ? Math.sqrt(Math.max(0, intBulge / rhoB0)) : 0;
    } else {
      sigmaBulgeGrid[i] = 0;
    }
  }

  return {
    ...analytic,
    vcSoftGrid,
    kappaGrid,
    sigmaHaloGrid,
    sigmaBulgeGrid,
    vEscGrid,
  };
}

export function evalDiskKinematics(profile: GalaxyRadialProfile, r: number): DiskKinematics {
  const p = profile.params;
  const rSafe = Math.max(0.15, r);
  const vc = evalProfileGrid(profile, profile.vcSoftGrid, rSafe);
  const kappa = evalProfileGrid(profile, profile.kappaGrid, rSafe);
  const vEsc = evalProfileGrid(profile, profile.vEscGrid, rSafe);

  const omega = vc / rSafe;
  const sigma0 = p.mDiskInf / (2.0 * Math.PI * p.rDisk * p.rDisk);
  const sigmaSurf = sigma0 * Math.exp(-rSafe / p.rDisk);
  const zScale = 2.0 * p.zDisk;

  const sigmaZ = Math.sqrt(Math.max(1.0, Math.PI * G * sigmaSurf * zScale));
  const qToomre = 1.45;
  const sigRToomre = (qToomre * 3.36 * G * sigmaSurf) / Math.max(1e-3, kappa);
  const sigmaR = Math.min(0.42 * vc, Math.max(sigRToomre, 1.35 * sigmaZ));
  const sigmaPhi = sigmaR * Math.min(1.0, kappa / Math.max(1e-3, 2.0 * omega));

  const driftTerm = sigmaR * sigmaR * (1.0 - (kappa * kappa) / Math.max(1e-4, 4.0 * omega * omega) - (2.0 * rSafe) / p.rDisk);
  const vPhiMean = Math.sqrt(Math.max(0, vc * vc + driftTerm));

  return { omega, vPhiMean, sigmaR, sigmaPhi, sigmaZ, vEsc };
}

export interface MacroSampleResult {
  readonly pos: Float32Array;     // [x, y, z, mass] * count
  readonly vel: Float32Array;     // [vx, vy, vz, epsSq] * count
  readonly compType: Uint8Array;  // 0 = halo, 1 = disk, 2 = bulge
  readonly eqProfile: GalaxyRadialProfile; // discrete softened equilibrium profile
}

export function sampleMacroGalaxy(
  profile: GalaxyRadialProfile,
  nHalo: number,
  nDisk: number,
  nBulge: number,
  basisView: Mat3,
  centerPosView: Vec3,
  centerVelView: Vec3,
  rng: RNG,
): MacroSampleResult {
  const p = profile.params;
  const total = nHalo + nDisk + nBulge;
  const pos = new Float32Array(total * 4);
  const vel = new Float32Array(total * 4);
  const compType = new Uint8Array(total);

  const mHaloEach = p.mVirHalo / nHalo;
  const mDiskEach = p.mDisk / nDisk;
  const mBulgeEach = p.mBulge / nBulge;

  const epsHSq = p.epsHalo * p.epsHalo;
  const epsDSq = p.epsDisk * p.epsDisk;
  const epsBSq = p.epsBulge * p.epsBulge;

  // Phase A: Sample all positions first
  let idx = 0;
  for (let i = 0; i < nHalo; i++) {
    const r = sampleHernquistRadius(rng, p.aHalo, p.fCutHalo);
    const [nx, ny, nz] = randomUnitVector(rng);
    const base = idx * 4;
    pos[base] = r * nx;
    pos[base + 1] = r * ny;
    pos[base + 2] = r * nz;
    pos[base + 3] = mHaloEach;
    vel[base + 3] = epsHSq;
    compType[idx++] = 0;
  }
  for (let i = 0; i < nDisk; i++) {
    const R = sampleExponentialRadius(rng, p.rDisk, p.fCutDisk);
    const z = sampleSech2Z(rng, p.zDisk);
    const phi = 2.0 * Math.PI * rng();
    const base = idx * 4;
    pos[base] = R * Math.cos(phi);
    pos[base + 1] = R * Math.sin(phi);
    pos[base + 2] = z;
    pos[base + 3] = mDiskEach;
    vel[base + 3] = epsDSq;
    compType[idx++] = 1;
  }
  for (let i = 0; i < nBulge; i++) {
    const r = sampleHernquistRadius(rng, p.aBulge, p.fCutBulge);
    const [nx, ny, nz] = randomUnitVector(rng);
    const base = idx * 4;
    pos[base] = r * nx;
    pos[base + 1] = r * ny;
    pos[base + 2] = r * nz;
    pos[base + 3] = mBulgeEach;
    vel[base + 3] = epsBSq;
    compType[idx++] = 2;
  }

  // Phase B: Calibrate softened Jeans & centrifugal profile against sampled discrete positions
  const eqProfile = buildSoftenedDiscreteProfile(profile, pos, vel, total);

  // Phase C: Sample equilibrium velocities
  idx = 0;
  for (let i = 0; i < nHalo; i++) {
    const base = idx * 4;
    const r = Math.hypot(pos[base], pos[base + 1], pos[base + 2]);
    const sig = evalProfileGrid(eqProfile, eqProfile.sigmaHaloGrid, r);
    const vEsc = evalProfileGrid(eqProfile, eqProfile.vEscGrid, r);
    let vx = gaussian(rng) * sig;
    let vy = gaussian(rng) * sig;
    let vz = gaussian(rng) * sig;
    const speed = Math.hypot(vx, vy, vz);
    if (speed > 0.95 * vEsc && speed > 1e-8) {
      const s = (0.95 * vEsc) / speed;
      vx *= s;
      vy *= s;
      vz *= s;
    }
    vel[base] = vx;
    vel[base + 1] = vy;
    vel[base + 2] = vz;
    idx++;
  }

  for (let i = 0; i < nDisk; i++) {
    const base = idx * 4;
    const x = pos[base];
    const y = pos[base + 1];
    const R = Math.hypot(x, y);
    const cosP = R > 1e-8 ? x / R : 1.0;
    const sinP = R > 1e-8 ? y / R : 0.0;

    const kin = evalDiskKinematics(eqProfile, R);
    const vR = gaussian(rng) * kin.sigmaR;
    const vPhi = kin.vPhiMean + gaussian(rng) * kin.sigmaPhi;
    const vz = gaussian(rng) * kin.sigmaZ;

    let vx = vR * cosP - vPhi * sinP;
    let vy = vR * sinP + vPhi * cosP;
    let vzClamped = vz;
    const speed = Math.hypot(vx, vy, vzClamped);
    if (speed > 0.95 * kin.vEsc && speed > 1e-8) {
      const s = (0.95 * kin.vEsc) / speed;
      vx *= s;
      vy *= s;
      vzClamped *= s;
    }
    vel[base] = vx;
    vel[base + 1] = vy;
    vel[base + 2] = vzClamped;
    idx++;
  }

  for (let i = 0; i < nBulge; i++) {
    const base = idx * 4;
    const x = pos[base];
    const y = pos[base + 1];
    const z = pos[base + 2];
    const r = Math.hypot(x, y, z);

    const sig = evalProfileGrid(eqProfile, eqProfile.sigmaBulgeGrid, r);
    const vc = evalProfileGrid(eqProfile, eqProfile.vcSoftGrid, r);
    const vEsc = evalProfileGrid(eqProfile, eqProfile.vEscGrid, r);

    const Rxy = Math.hypot(x, y);
    const vRot = 0.22 * vc;
    const rotVx = Rxy > 1e-5 ? (-vRot * y) / Rxy : 0;
    const rotVy = Rxy > 1e-5 ? (vRot * x) / Rxy : 0;

    let vx = rotVx + gaussian(rng) * sig;
    let vy = rotVy + gaussian(rng) * sig;
    let vz = gaussian(rng) * sig;
    const speed = Math.hypot(vx, vy, vz);
    if (speed > 0.95 * vEsc && speed > 1e-8) {
      const s = (0.95 * vEsc) / speed;
      vx *= s;
      vy *= s;
      vz *= s;
    }
    vel[base] = vx;
    vel[base + 1] = vy;
    vel[base + 2] = vz;
    idx++;
  }

  // Exact COM & zero-momentum recentering in local frame before rotating & boosting
  let totalMass = 0;
  let cmX = 0;
  let cmY = 0;
  let cmZ = 0;
  let pmX = 0;
  let pmY = 0;
  let pmZ = 0;
  for (let i = 0; i < total; i++) {
    const base = i * 4;
    const m = pos[base + 3];
    totalMass += m;
    cmX += m * pos[base];
    cmY += m * pos[base + 1];
    cmZ += m * pos[base + 2];
    pmX += m * vel[base];
    pmY += m * vel[base + 1];
    pmZ += m * vel[base + 2];
  }
  cmX /= totalMass;
  cmY /= totalMass;
  cmZ /= totalMass;
  const vcmX = pmX / totalMass;
  const vcmY = pmY / totalMass;
  const vcmZ = pmZ / totalMass;

  for (let i = 0; i < total; i++) {
    const base = i * 4;
    const pView = applyMat3(basisView, [
      pos[base] - cmX,
      pos[base + 1] - cmY,
      pos[base + 2] - cmZ,
    ]);
    const vView = applyMat3(basisView, [
      vel[base] - vcmX,
      vel[base + 1] - vcmY,
      vel[base + 2] - vcmZ,
    ]);

    pos[base] = pView[0] + centerPosView[0];
    pos[base + 1] = pView[1] + centerPosView[1];
    pos[base + 2] = pView[2] + centerPosView[2];

    vel[base] = vView[0] + centerVelView[0];
    vel[base + 1] = vView[1] + centerVelView[1];
    vel[base + 2] = vView[2] + centerVelView[2];
  }

  return { pos, vel, compType, eqProfile };
}

export function getCircularSpeedKms(profile: GalaxyRadialProfile, rKpc: number): number {
  return evalProfileGrid(profile, profile.vcTrueGrid, rKpc) * KPC_PER_GYR_TO_KMS;
}
