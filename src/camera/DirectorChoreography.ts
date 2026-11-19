import * as THREE from 'three';
import { M31_PARAMS, MW_PARAMS, Vec3 } from '../physics/constants.ts';

const MW_MASS_FRAC = MW_PARAMS.mTotal / (MW_PARAMS.mTotal + M31_PARAMS.mTotal);
const M31_MASS_FRAC = 1.0 - MW_MASS_FRAC;

export interface DirectorPose {
  readonly position: THREE.Vector3;
  readonly target: THREE.Vector3;
  fov: number;
  rollRad: number;
}

interface DirectorKeyframe {
  readonly t: number;
  readonly mwFocusWeight: number; // 1.0 = over-the-rim MW shot, 0.0 = system COM
  readonly azimDeg: number;       // 0 = +Z broadside, negative = sweeping behind MW (-X)
  readonly elevDeg: number;       // elevation above XZ orbital plane
  readonly baseDist: number;      // minimum camera distance (kpc)
  readonly sepDistMult: number;   // additional distance proportional to separation (kpc)
  readonly fovDeg: number;        // dynamic lens focal length (deg)
  readonly rollDeg: number;       // Interstellar-style subtle camera roll bank (deg)
  readonly screenLift: number;    // subtle upward framing shift above bottom timeline bar
}

// Location-scouted Full-3D 7-Act Interstellar choreography (416-deg continuous 3D corkscrew):
//   Act I   (0.00..2.60): Open MW spiral foreground (-98,-15) sweeping up to M31 loom (-42,+26)
//   Act II  (3.12..3.55): Side-by-side full spirals opening (+8,+34)
//   Act III (3.82):       First Pericenter (+32,+44) 97% face-on MW + tilted M31 collision
//   Act IV  (4.10..4.43): Widescreen Tidal Sweep (+58,+54) perpendicular to sepAzim=-36 deg
//   Act V   (4.90..5.29): Razor equatorial dive (+116,+8) into southern underbelly (+178,-44)
//   Act VI  (5.76..6.35): Wide southern 3D arc around merging nuclei & outer shells
//   Act VII (7.50):       Ascension over reborn Milkomeda (+318,+32, 355 kpc)
const KEYFRAMES: DirectorKeyframe[] = [
  { t: 0.00, mwFocusWeight: 1.00, azimDeg:  -98, elevDeg: -15, baseDist: 148, sepDistMult: 0.00, fovDeg: 38.0, rollDeg:   6.0, screenLift: 0.020 },
  { t: 1.60, mwFocusWeight: 0.72, azimDeg:  -68, elevDeg:  12, baseDist: 175, sepDistMult: 0.06, fovDeg: 40.0, rollDeg:   4.0, screenLift: 0.024 },
  { t: 2.60, mwFocusWeight: 0.15, azimDeg:  -42, elevDeg:  26, baseDist: 140, sepDistMult: 0.55, fovDeg: 41.0, rollDeg:   3.0, screenLift: 0.028 },
  { t: 3.12, mwFocusWeight: 0.00, azimDeg:  -18, elevDeg:  30, baseDist:  42, sepDistMult: 0.96, fovDeg: 42.0, rollDeg:   2.0, screenLift: 0.030 },
  { t: 3.55, mwFocusWeight: 0.00, azimDeg:    8, elevDeg:  34, baseDist:  78, sepDistMult: 0.90, fovDeg: 40.0, rollDeg:  -4.0, screenLift: 0.028 },
  { t: 3.82, mwFocusWeight: 0.00, azimDeg:   32, elevDeg:  44, baseDist: 108, sepDistMult: 1.00, fovDeg: 37.0, rollDeg:  -9.0, screenLift: 0.026 },
  { t: 4.10, mwFocusWeight: 0.00, azimDeg:   48, elevDeg:  50, baseDist: 105, sepDistMult: 1.05, fovDeg: 43.0, rollDeg:  -2.0, screenLift: 0.030 },
  { t: 4.43, mwFocusWeight: 0.00, azimDeg:   58, elevDeg:  54, baseDist:  92, sepDistMult: 1.12, fovDeg: 46.0, rollDeg:   4.0, screenLift: 0.030 },
  { t: 4.90, mwFocusWeight: 0.00, azimDeg:  116, elevDeg:   8, baseDist: 110, sepDistMult: 0.90, fovDeg: 42.0, rollDeg: -13.0, screenLift: 0.028 },
  { t: 5.29, mwFocusWeight: 0.00, azimDeg:  178, elevDeg: -44, baseDist: 175, sepDistMult: 0.90, fovDeg: 40.0, rollDeg: -10.0, screenLift: 0.028 },
  { t: 5.76, mwFocusWeight: 0.00, azimDeg:  218, elevDeg: -34, baseDist: 255, sepDistMult: 0.55, fovDeg: 45.0, rollDeg:   8.0, screenLift: 0.030 },
  { t: 6.35, mwFocusWeight: 0.00, azimDeg:  262, elevDeg:  -6, baseDist: 310, sepDistMult: 0.20, fovDeg: 46.0, rollDeg:   4.0, screenLift: 0.030 },
  { t: 7.50, mwFocusWeight: 0.00, azimDeg:  318, elevDeg:  32, baseDist: 355, sepDistMult: 0.00, fovDeg: 47.0, rollDeg:   0.0, screenLift: 0.032 },
];

type TrackKey = Exclude<keyof DirectorKeyframe, 't'>;

interface PchipSpline {
  readonly t: Float64Array;
  readonly y: Float64Array;
  readonly m: Float64Array;
}

// Fritsch-Carlson monotone cubic Hermite spline (C^1 continuous, zero overshoot)
function buildPchipSpline(key: TrackKey): PchipSpline {
  const n = KEYFRAMES.length;
  const t = new Float64Array(n);
  const y = new Float64Array(n);
  const m = new Float64Array(n);
  const h = new Float64Array(n - 1);
  const delta = new Float64Array(n - 1);

  for (let i = 0; i < n; i++) {
    t[i] = KEYFRAMES[i].t;
    y[i] = KEYFRAMES[i][key];
  }
  for (let i = 0; i < n - 1; i++) {
    h[i] = t[i + 1] - t[i];
    delta[i] = (y[i + 1] - y[i]) / h[i];
  }

  m[0] = delta[0];
  m[n - 1] = delta[n - 2];

  for (let i = 1; i < n - 1; i++) {
    if (delta[i - 1] * delta[i] <= 0) {
      m[i] = 0;
    } else {
      const w1 = 2 * h[i] + h[i - 1];
      const w2 = h[i] + 2 * h[i - 1];
      m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
    }
  }

  for (let i = 0; i < n - 1; i++) {
    if (Math.abs(delta[i]) < 1e-9) {
      m[i] = 0;
      m[i + 1] = 0;
    } else {
      const a = m[i] / delta[i];
      const b = m[i + 1] / delta[i];
      const r2 = a * a + b * b;
      if (r2 > 9.0) {
        const tau = 3.0 / Math.sqrt(r2);
        m[i] = tau * a * delta[i];
        m[i + 1] = tau * b * delta[i];
      }
    }
  }

  return { t, y, m };
}

function evalSpline(sp: PchipSpline, x: number): number {
  const n = sp.t.length;
  if (x <= sp.t[0]) return sp.y[0];
  if (x >= sp.t[n - 1]) return sp.y[n - 1];

  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sp.t[mid] <= x) lo = mid;
    else hi = mid;
  }

  const h = sp.t[lo + 1] - sp.t[lo];
  const s = (x - sp.t[lo]) / h;
  const s2 = s * s;
  const s3 = s2 * s;

  const h00 = 2 * s3 - 3 * s2 + 1;
  const h10 = s3 - 2 * s2 + s;
  const h01 = -2 * s3 + 3 * s2;
  const h11 = s3 - s2;

  return h00 * sp.y[lo] + h10 * h * sp.m[lo] + h01 * sp.y[lo + 1] + h11 * h * sp.m[lo + 1];
}

const SPLINES = {
  mwFocusWeight: buildPchipSpline('mwFocusWeight'),
  azimDeg: buildPchipSpline('azimDeg'),
  elevDeg: buildPchipSpline('elevDeg'),
  baseDist: buildPchipSpline('baseDist'),
  sepDistMult: buildPchipSpline('sepDistMult'),
  fovDeg: buildPchipSpline('fovDeg'),
  rollDeg: buildPchipSpline('rollDeg'),
  screenLift: buildPchipSpline('screenLift'),
};

const _mwVec = new THREE.Vector3();
const _m31Vec = new THREE.Vector3();
const _comVec = new THREE.Vector3();
const _dirToM31 = new THREE.Vector3();
const _mwRimAnchor = new THREE.Vector3();
const _viewDir = new THREE.Vector3();
const _camRight = new THREE.Vector3();
const _camUp = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);

export function evaluateDirectorShot(
  timeGyr: number,
  mwCenter: Vec3,
  m31Center: Vec3,
  sepKpc: number,
  realTimeSec: number,
  outPose: DirectorPose,
): void {
  const mwW = Math.max(0, Math.min(1, evalSpline(SPLINES.mwFocusWeight, timeGyr)));
  const azimBaseDeg = evalSpline(SPLINES.azimDeg, timeGyr);
  const elevBaseDeg = evalSpline(SPLINES.elevDeg, timeGyr);
  const baseDist = evalSpline(SPLINES.baseDist, timeGyr);
  const sepMult = evalSpline(SPLINES.sepDistMult, timeGyr);
  const fovDeg = evalSpline(SPLINES.fovDeg, timeGyr);
  const rollDeg = evalSpline(SPLINES.rollDeg, timeGyr);
  const screenLift = evalSpline(SPLINES.screenLift, timeGyr);

  _mwVec.set(mwCenter[0], mwCenter[1], mwCenter[2]);
  _m31Vec.set(m31Center[0], m31Center[1], m31Center[2]);
  _comVec.set(
    MW_MASS_FRAC * mwCenter[0] + M31_MASS_FRAC * m31Center[0],
    MW_MASS_FRAC * mwCenter[1] + M31_MASS_FRAC * m31Center[1],
    MW_MASS_FRAC * mwCenter[2] + M31_MASS_FRAC * m31Center[2],
  );

  // Over-the-rim MW anchor gently leads MW toward M31 so both frame cleanly in Act I
  _dirToM31.subVectors(_m31Vec, _mwVec);
  const len = _dirToM31.length();
  if (len > 1e-4) _dirToM31.multiplyScalar(1.0 / len);
  else _dirToM31.set(1, 0, 0);
  _mwRimAnchor.copy(_mwVec).addScaledVector(_dirToM31, 20.0);

  outPose.target.copy(_comVec).lerp(_mwRimAnchor, mwW);

  const dist = Math.max(baseDist, baseDist + sepMult * sepKpc);

  // Ultra-subtle weighted zero-G breathing sway so even paused scenes feel alive
  const swayAzim = Math.sin(realTimeSec * 0.17) * 0.011;
  const swayElev = Math.sin(realTimeSec * 0.23 + 1.1) * 0.006;

  const azim = THREE.MathUtils.degToRad(azimBaseDeg) + swayAzim;
  const elev = THREE.MathUtils.degToRad(elevBaseDeg) + swayElev;
  const cosE = Math.cos(elev);

  const ox = Math.sin(azim) * cosE * dist;
  const oy = Math.sin(elev) * dist;
  const oz = Math.cos(azim) * cosE * dist;

  outPose.position.set(
    outPose.target.x + ox,
    outPose.target.y + oy,
    outPose.target.z + oz,
  );

  // Shift look target slightly downward along camera-up so galaxies sit cleanly above bottom HUD
  _viewDir.subVectors(outPose.target, outPose.position).normalize();
  _camRight.crossVectors(_viewDir, _worldUp).normalize();
  _camUp.crossVectors(_camRight, _viewDir).normalize();
  outPose.target.addScaledVector(_camUp, -dist * screenLift);

  outPose.fov = fovDeg;
  outPose.rollRad = THREE.MathUtils.degToRad(rollDeg);
}
