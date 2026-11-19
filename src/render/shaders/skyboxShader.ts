export const SKYBOX_VERT = /* glsl */ `
precision highp float;

in vec3 position;
out vec2 vNdc;

void main() {
  vNdc = position.xy;
  gl_Position = vec4(position.xy, 0.9999, 1.0);
}
`;

export const SKYBOX_FRAG = /* glsl */ `
precision highp float;

in vec2 vNdc;

uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uTanHalfFov;
uniform float uAspect;

out vec4 fragColor;

float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}

vec3 hash33(vec3 p) {
  p = vec3(
    dot(p, vec3(127.1, 311.7, 74.7)),
    dot(p, vec3(269.5, 183.3, 246.1)),
    dot(p, vec3(113.5, 271.9, 124.6))
  );
  return fract(sin(p) * 43758.5453123);
}

float noise3(vec3 x) {
  vec3 i = floor(x);
  vec3 f = fract(x);
  f = f * f * (3.0 - 2.0 * f);

  return mix(
    mix(
      mix(hash13(i + vec3(0, 0, 0)), hash13(i + vec3(1, 0, 0)), f.x),
      mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x),
      f.y
    ),
    mix(
      mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x),
      mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x),
      f.y
    ),
    f.z
  );
}

float fbm3(vec3 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 4; i++) {
    v += a * noise3(p);
    p = p * 2.07 + vec3(13.1, 7.7, 19.3);
    a *= 0.5;
  }
  return v;
}

vec3 starColor(float temp) {
  vec3 cool = vec3(1.0, 0.58, 0.30);
  vec3 mid  = vec3(1.0, 0.94, 0.86);
  vec3 hot  = vec3(0.46, 0.74, 1.0);
  return temp < 0.5
    ? mix(cool, mid, temp * 2.0)
    : mix(mid, hot, (temp - 0.5) * 2.0);
}

vec3 evalStarLayer(vec3 dir, float scale, float density, float brightness) {
  vec3 p = dir * scale;
  vec3 id = floor(p);
  vec3 f = fract(p) - 0.5;
  vec3 rnd = hash33(id);

  if (rnd.x > density) return vec3(0.0);

  vec3 jitter = (rnd - 0.5) * 0.65;
  float d = length(f - jitter);
  float size = mix(0.016, 0.044, pow(rnd.y, 4.0));
  float spot = exp(-(d * d) / (size * size));
  float mag = pow(rnd.z, 4.5) * brightness;
  return starColor(fract(rnd.x * 17.3)) * spot * mag;
}

// Rare bright foreground guide stars with Airy cores and hairline 4-point diffraction spikes
vec3 evalHeroGuideStars(vec3 dir) {
  vec3 p = dir * 28.0;
  vec3 id = floor(p);
  vec3 f = fract(p) - 0.5;
  vec3 rnd = hash33(id + 19.7);

  if (rnd.x > 0.016) return vec3(0.0);

  // Strict max radius <= 0.16 inside [-0.5, 0.5]^3 cell with jitter <= 0.14
  // guarantees 0.14 + 0.16 = 0.30 < 0.50 -> mathematically zero cell-wall clipping!
  vec3 jitter = (rnd - 0.5) * 0.28;
  vec3 rel = f - jitter;

  vec3 t1 = normalize(cross(dir, abs(dir.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 t2 = cross(dir, t1);
  float u = dot(rel, t1);
  float v = dot(rel, t2);

  float d = length(vec2(u, v));
  if (d > 0.16) return vec3(0.0);

  float ang = 0.785398 + (rnd.y - 0.5) * 0.22;
  float ca = cos(ang);
  float sa = sin(ang);
  vec2 rot = vec2(ca * u - sa * v, sa * u + ca * v);

  float core = exp(-(d * d) / (0.010 * 0.010)) * 2.1;
  float halo = exp(-d * 34.0) * 0.14;
  float spike1 = exp(-abs(rot.x) * 320.0) * exp(-abs(rot.y) * 32.0);
  float spike2 = exp(-abs(rot.y) * 320.0) * exp(-abs(rot.x) * 32.0);
  float spikes = (spike1 + spike2) * 0.55 * smoothstep(0.16, 0.02, d);

  float mag = 0.40 + 0.65 * pow(rnd.z, 2.0);
  vec3 col = starColor(fract(rnd.x * 41.3 + rnd.y));
  return col * (core + halo + spikes) * mag;
}

vec3 evalMicroGalaxies(vec3 dir, float scale, float threshold, float brightness, float seedOff) {
  vec3 p = dir * scale;
  vec3 id = floor(p);
  vec3 f = fract(p) - 0.5;
  vec3 rnd = hash33(id + seedOff);

  if (rnd.x > threshold) return vec3(0.0);

  vec3 center = (rnd - 0.5) * 0.48;
  vec3 rel = f - center;

  vec3 t1 = normalize(cross(dir, abs(dir.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 t2 = cross(dir, t1);
  float u = dot(rel, t1);
  float v = dot(rel, t2);

  float ang = rnd.y * 6.283185;
  float ca = cos(ang);
  float sa = sin(ang);
  vec2 rot = vec2(ca * u - sa * v, sa * u + ca * v);
  float tilt = mix(1.4, 3.6, rnd.z);
  float r = length(vec2(rot.x, rot.y * tilt));

  float disk = exp(-r * 52.0) * 0.045;
  float core = exp(-r * 135.0) * 0.11;
  // JWST deep-field mix: amber/redshifted ellipticals, warm ivory, and faint blue irregulars
  float tint = fract(rnd.x * 53.7);
  vec3 col = tint < 0.45
    ? vec3(1.0, 0.62, 0.30)
    : (tint < 0.78 ? vec3(1.0, 0.86, 0.66) : vec3(0.56, 0.78, 1.0));
  return col * (disk + core) * brightness * smoothstep(0.14, 0.015, r);
}

// Handcrafted distant spiral galaxy landmark on the sky sphere
vec3 evalHeroSpiral(
  vec3 dir,
  vec3 galDir,
  float radius,
  float tilt,
  float rotAngle,
  float brightness
) {
  float cosA = dot(dir, galDir);
  if (cosA < 0.88) return vec3(0.0);

  vec3 t1 = normalize(cross(galDir, abs(galDir.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 t2 = cross(galDir, t1);
  float u = dot(dir, t1);
  float v = dot(dir, t2);

  float ca = cos(rotAngle);
  float sa = sin(rotAngle);
  vec2 rot = vec2(ca * u - sa * v, sa * u + ca * v);
  vec2 ell = vec2(rot.x, rot.y * tilt) / radius;
  float r = length(ell);
  if (r > 1.45) return vec3(0.0);

  float theta = atan(ell.y, ell.x);
  float armPhase = theta + 3.2 * log(max(r, 0.06));
  float arms = pow(0.5 + 0.5 * cos(2.0 * armPhase), 2.1);
  float fineTex = 0.72 + 0.56 * noise3(vec3(ell * 5.2, 3.1));

  float bulge = exp(-r * 5.6) * 1.15;
  float disk = exp(-r * 2.2) * (0.28 + 0.72 * arms) * fineTex;
  float mask = smoothstep(1.4, 0.3, r);

  vec3 bulgeCol = vec3(1.0, 0.76, 0.44);
  vec3 armCol   = vec3(0.42, 0.70, 1.0);
  return (bulgeCol * bulge + armCol * disk * 0.55) * brightness * mask;
}

// Handcrafted distant edge-on Sombrero-style galaxy with dark equatorial dust lane
vec3 evalHeroEdgeOn(
  vec3 dir,
  vec3 galDir,
  float radius,
  float rotAngle,
  float brightness
) {
  float cosA = dot(dir, galDir);
  if (cosA < 0.88) return vec3(0.0);

  vec3 t1 = normalize(cross(galDir, abs(galDir.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  vec3 t2 = cross(galDir, t1);
  float u = dot(dir, t1);
  float v = dot(dir, t2);

  float ca = cos(rotAngle);
  float sa = sin(rotAngle);
  vec2 rot = vec2(ca * u - sa * v, sa * u + ca * v) / radius;
  float rBulge = length(rot * vec2(1.15, 2.2));
  float rDisk  = length(rot * vec2(1.0, 6.2));
  if (rDisk > 1.5) return vec3(0.0);

  float bulge = exp(-rBulge * 4.8) * 1.1;
  float disk = exp(-rDisk * 3.4) * 0.52 * smoothstep(1.45, 0.15, rDisk);
  float dustLane = 1.0 - 0.85 * exp(-pow(rot.y / 0.052, 2.0)) * smoothstep(1.05, 0.12, abs(rot.x));

  vec3 bulgeCol = vec3(1.0, 0.76, 0.44);
  vec3 diskCol  = vec3(0.92, 0.72, 0.52);
  return (bulgeCol * bulge + diskCol * disk) * dustLane * brightness;
}

void main() {
  vec3 dir = normalize(
    uCamForward +
    uCamRight * (vNdc.x * uAspect * uTanHalfFov) +
    uCamUp * (vNdc.y * uTanHalfFov)
  );

  // 1. True obsidian deep-space void floor
  vec3 bg = vec3(0.00009, 0.00014, 0.00036);

  // 2. Whisper-subtle sculpted domain-warped cosmic dust band + dark absorption rifts
  vec3 bandNormal = normalize(vec3(0.32, 0.88, -0.35));
  float bandDist = dot(dir, bandNormal);
  float bandEnv = exp(-bandDist * bandDist * 7.5);

  float q = fbm3(dir * 2.4 + vec3(1.7, -2.3, 4.1));
  float w = fbm3(dir * 3.8 + vec3(1.6 * q, -1.2 * q, 0.9 * q));
  float fineWisp = fbm3(dir * 8.6 + vec3(w * 1.4));
  float riftNoise = fbm3(dir * 6.2 + vec3(-3.4, 2.1, 5.7) + vec3(0.8 * q));
  float dustRift = 1.0 - 0.82 * smoothstep(0.40, 0.70, riftNoise);

  vec3 cobaltCloud = vec3(0.0010, 0.0019, 0.0044) * smoothstep(0.30, 0.82, w);
  vec3 tealCloud   = vec3(0.0005, 0.0018, 0.0028) * smoothstep(0.45, 0.88, fineWisp);
  vec3 copperWisp  = vec3(0.0020, 0.0008, 0.0011) * smoothstep(0.50, 0.86, q) * (1.0 - dustRift * 0.35);

  bg += (cobaltCloud + tealCloud + copperWisp) * (0.14 + 0.86 * bandEnv) * dustRift;

  // 3. Localized off-axis deep-space emission nebula accent (upper-right quadrant)
  vec3 nebDir = normalize(vec3(0.20, -0.20, -0.96));
  float nebMask = pow(max(0.0, dot(dir, nebDir)), 7.5);
  float nebTex = fbm3(dir * 6.5 + vec3(4.2, -1.8, 2.9) + vec3(1.2 * q));
  float nebCore = smoothstep(0.38, 0.84, nebTex) * dustRift;
  vec3 nebHalpha = vec3(0.0068, 0.0014, 0.0032);
  vec3 nebOIII   = vec3(0.0012, 0.0045, 0.0062);
  bg += mix(nebOIII, nebHalpha, smoothstep(0.35, 0.72, fineWisp)) * nebMask * nebCore;

  // 4. 4 Handcrafted distant deep-field hero galaxies (small & subtle)
  bg += evalHeroSpiral(dir, normalize(vec3(-0.31, -0.15, -0.94)), 0.020, 2.05, 0.42, 0.16);
  bg += evalHeroEdgeOn(dir, normalize(vec3( 0.34, -0.52, -0.89)), 0.018, -0.35, 0.18);
  bg += evalHeroSpiral(dir, normalize(vec3( 0.42,  0.32,  0.85)), 0.019, 2.20, -0.75, 0.15);
  bg += evalHeroSpiral(dir, normalize(vec3(-0.55, -0.28,  0.78)), 0.017, 1.65, 1.10, 0.14);

  // 5. Multi-scale JWST Ultra-Deep Field micro-galaxies
  bg += evalMicroGalaxies(dir, 22.0, 0.050, 0.85, 71.9);
  bg += evalMicroGalaxies(dir, 42.0, 0.048, 0.48, 113.3);

  // 6. 4-tier spectral starfield (band-enriched) + rare diffraction-spiked guide stars
  float starBoost = 1.0 + 0.55 * bandEnv * dustRift;
  bg += evalStarLayer(dir, 540.0, 0.48, 0.52) * starBoost;
  bg += evalStarLayer(dir, 280.0, 0.38, 1.15) * starBoost;
  bg += evalStarLayer(dir, 150.0, 0.25, 0.75);
  bg += evalHeroGuideStars(dir);

  fragColor = vec4(bg, 1.0);
}
`;
