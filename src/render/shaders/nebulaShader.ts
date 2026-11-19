export const NEBULA_VERT = /* glsl */ `
precision highp float;
precision highp int;

in vec3 position;

uniform sampler2D uPosTex;
uniform sampler2D uVelTex;
uniform sampler2D uStaticAttrTex;
uniform sampler2D uStaticColorTex;

uniform float uFocalLength;
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

out vec3 vColor;
out float vAlpha;
out float vSeed;
out float vAngle;
out float vStretch;
out float vIsHII;

void main() {
  int idx = int(position.x + 0.5);
  ivec2 coord = ivec2(idx & 511, idx >> 9);

  vec4 posData = texelFetch(uPosTex, coord, 0);
  vec4 velData = texelFetch(uVelTex, coord, 0);
  vec4 attr = texelFetch(uStaticAttrTex, coord, 0);
  vec4 colData = texelFetch(uStaticColorTex, coord, 0);

  vec4 mvPos = uModelViewMatrix * vec4(posData.xyz, 1.0);
  float dist = -mvPos.z;
  if (dist <= 0.25) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    vAlpha = 0.0;
    return;
  }

  gl_Position = uProjectionMatrix * mvPos;

  float baseSizeKpc = attr.y;
  float lum = attr.z;
  float starburst = posData.w;
  int packedType = int(attr.x + 0.5);
  int popType = packedType - (packedType / 10) * 10;

  float projPx = (baseSizeKpc * uFocalLength) / dist;
  gl_PointSize = clamp(projPx, 2.5, 195.0);

  // Smooth fade when camera flies right inside a cloud
  float nearFade = smoothstep(3.0, 14.0, dist);
  // Energy conservation if cloud expands beyond clamp or becomes tiny
  float subScale = clamp(projPx / gl_PointSize, 0.25, 1.0);

  // Project 3D velocity onto camera screen plane to stretch gas along tidal streamlines
  vec3 mvVel = (uModelViewMatrix * vec4(velData.xyz, 0.0)).xyz;
  float velMag2D = length(mvVel.xy);
  vAngle = velMag2D > 1e-3 ? atan(mvVel.y, mvVel.x) : attr.w * 6.283185;
  vStretch = clamp(velMag2D / 300.0, 0.0, 0.65);

  vec3 col = colData.rgb;
  if (popType == 5) {
    // H-II emission nebula: surges vivid crimson/cyan during tidal shock starburst
    col = mix(col, vec3(1.0, 0.14, 0.46), clamp(starburst * 0.45, 0.0, 0.65));
    vAlpha = nearFade * subScale * lum * (0.045 + 0.11 * starburst);
    vIsHII = 1.0;
  } else {
    // Soft volumetric gas / unresolved stellar continuum
    col = mix(col, vec3(0.28, 0.64, 1.0), clamp(starburst * 0.45, 0.0, 0.65));
    vAlpha = nearFade * subScale * lum * (0.011 + 0.022 * starburst);
    vIsHII = 0.0;
  }

  vColor = col;
  vSeed = attr.w * 43.7;
}
`;

export const NEBULA_FRAG = /* glsl */ `
precision highp float;

in vec3 vColor;
in float vAlpha;
in float vSeed;
in float vAngle;
in float vStretch;
in float vIsHII;

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

float wispyFbm(vec2 p) {
  float v = 0.0;
  v += 0.65 * noise2(p);
  p = mat2(0.8, -0.6, 0.6, 0.8) * p * 2.15 + vec2(3.7, 1.9);
  v += 0.35 * noise2(p);
  return v;
}

void main() {
  vec2 uv = gl_PointCoord - 0.5;
  float rawR = length(uv);
  if (rawR > 0.5 || vAlpha <= 0.0004) discard;

  float ca = cos(vAngle);
  float sa = sin(vAngle);
  vec2 rot = vec2(ca * uv.x + sa * uv.y, -sa * uv.x + ca * uv.y);
  vec2 stretched = vec2(rot.x / (1.0 + vStretch), rot.y * (1.0 + 0.38 * vStretch));

  float r2 = dot(stretched, stretched);
  float radialSoft = pow(max(0.0, 1.0 - rawR * 2.0), 2.4);
  float falloff = vIsHII > 0.5 ? 22.0 : 14.5;
  float env = exp(-r2 * falloff) * radialSoft;

  float wisp = wispyFbm(stretched * 5.2 + vec2(vSeed, vSeed * 0.63));
  float density = env * (0.40 + 0.90 * wisp);

  fragColor = vec4(vColor * (vAlpha * density), 1.0);
}
`;
