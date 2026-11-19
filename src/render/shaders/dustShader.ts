export const DUST_VERT = /* glsl */ `
precision highp float;
precision highp int;

in vec3 position;

uniform sampler2D uPosTex;
uniform sampler2D uVelTex;
uniform sampler2D uStaticAttrTex;

uniform vec3 uMwCenter;
uniform vec3 uM31Center;
uniform vec3 uCamForward;
uniform float uFocalLength;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

out float vOpacity;
out float vSeed;
out float vAngle;
out float vStretch;

void main() {
  int idx = int(position.x + 0.5);
  ivec2 coord = ivec2(idx & 511, idx >> 9);

  vec4 posData = texelFetch(uPosTex, coord, 0);
  vec4 velData = texelFetch(uVelTex, coord, 0);
  vec4 attr = texelFetch(uStaticAttrTex, coord, 0);

  vec4 mvPos = uModelViewMatrix * vec4(posData.xyz, 1.0);
  float dist = -mvPos.z;
  if (dist <= 0.25) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    vOpacity = 0.0;
    return;
  }

  gl_Position = uProjectionMatrix * mvPos;

  float baseSizeKpc = attr.y;
  float weight = attr.z;

  float projPx = (baseSizeKpc * uFocalLength) / dist;
  gl_PointSize = clamp(projPx, 2.0, 68.0);

  float nearFade = smoothstep(3.5, 16.0, dist);
  float closeSoften = mix(0.45, 1.0, smoothstep(20.0, 130.0, dist));
  float subScale = clamp(projPx / gl_PointSize, 0.25, 1.0);

  // Soften far-side dust passing behind the bright golden bulge so near-side
  // dust lanes silhouette crisply across the luminous galactic core
  int galId = int(attr.x + 0.5) / 10;
  vec3 hostCenter = (galId == 0) ? uMwCenter : uM31Center;
  vec3 relVec = posData.xyz - hostCenter;
  float relDepth = dot(relVec, uCamForward);
  float coreDist = length(relVec);
  float farSideSoften = (relDepth > 0.0)
    ? (0.45 + 0.55 * smoothstep(2.0, 9.0, coreDist))
    : 1.0;

  // Attenuate dust clouds flung far into sparse tidal voids so they don't speckle outer streams
  float minGalDist = min(length(posData.xyz - uMwCenter), length(posData.xyz - uM31Center));
  float diskMask = smoothstep(55.0, 18.0, minGalDist);

  vec3 mvVel = (uModelViewMatrix * vec4(velData.xyz, 0.0)).xyz;
  float velMag2D = length(mvVel.xy);
  vAngle = velMag2D > 1e-3 ? atan(mvVel.y, mvVel.x) : attr.w * 6.283185;
  vStretch = clamp(velMag2D / 290.0, 0.0, 0.72);

  vOpacity = clamp(weight * nearFade * closeSoften * subScale * farSideSoften * diskMask * 0.19, 0.0, 0.48);
  vSeed = attr.w * 59.3;
}
`;

export const DUST_FRAG = /* glsl */ `
precision highp float;

in float vOpacity;
in float vSeed;
in float vAngle;
in float vStretch;

out vec4 fragColor;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

float noise2(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash12(i), hash12(i + vec2(1.0, 0.0)), u.x),
    mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), u.x),
    u.y
  );
}

float filamentFbm(vec2 p) {
  float v = 0.0;
  v += 0.62 * noise2(p);
  p = mat2(0.75, -0.66, 0.66, 0.75) * p * 2.2 + vec2(5.1, 2.3);
  v += 0.38 * noise2(p);
  return v;
}

void main() {
  if (vOpacity <= 0.001) discard;

  vec2 uv = gl_PointCoord - 0.5;
  float rawR = length(uv);
  if (rawR > 0.5) discard;

  float ca = cos(vAngle);
  float sa = sin(vAngle);
  vec2 rot = vec2(ca * uv.x + sa * uv.y, -sa * uv.x + ca * uv.y);
  vec2 stretched = vec2(rot.x / (1.0 + vStretch), rot.y * (1.0 + 0.42 * vStretch));

  float r2 = dot(stretched, stretched);
  float radialSoft = pow(max(0.0, 1.0 - rawR * 2.0), 2.0);
  float env = exp(-r2 * 14.0) * radialSoft;
  float wisp = filamentFbm(stretched * 5.8 + vec2(vSeed, vSeed * 0.47));
  float a = clamp(vOpacity * env * (0.42 + 0.92 * wisp), 0.0, 0.55);

  // Cardelli-Clayton-Mathis R_V=3.1 wavelength-dependent Beer-Lambert extinction:
  // dst_new = dst * (1 - src.rgb). Absorbs blue strongest, red weakest -> warm amber reddening!
  fragColor = vec4(0.42 * a, 0.76 * a, 1.0 * a, 1.0);
}
`;
