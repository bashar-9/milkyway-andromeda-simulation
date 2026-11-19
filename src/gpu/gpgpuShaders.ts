export const GPGPU_QUAD_VERT = /* glsl */ `
precision highp float;

in vec3 position;
in vec2 uv;
out vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const GPGPU_ADVECT_FRAG = /* glsl */ `
precision highp float;
precision highp int;

uniform sampler2D uPosTex;
uniform sampler2D uVelTex;
uniform sampler2D uStaticAttrTex;
uniform sampler2D uTreePosMass;
uniform sampler2D uTreeMeta;

uniform int uNodeCount;
uniform int uIsFirstStep;
uniform float uG;
uniform float uThetaSq;
uniform float uKickDt;
uniform float uDriftDt;
uniform float uStepDt;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outVel;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec4 posData = texelFetch(uPosTex, coord, 0);
  vec4 velData = texelFetch(uVelTex, coord, 0);
  vec4 attr = texelFetch(uStaticAttrTex, coord, 0);

  vec3 pos = posData.xyz;
  float starburst = posData.w;
  vec3 vel = velData.xyz;
  float baselineDensity = velData.w;

  int packedType = int(attr.x + 0.5);
  int popType = packedType - (packedType / 10) * 10;

  // Match macro softening lengths: bulge=0.65, halo=5.5, disk/gas/dust=1.2 kpc
  float selfEps = (popType == 2) ? 0.65 : ((popType == 3) ? 5.5 : 1.2);
  float selfEpsSq = selfEps * selfEps;

  vec3 acc = vec3(0.0);
  float localDensity = 0.0;

  int node = 0;
  for (int iter = 0; iter < 1536; iter++) {
    if (node >= uNodeCount) break;

    ivec2 tCoord = ivec2(node & 127, node >> 7);
    vec4 pm = texelFetch(uTreePosMass, tCoord, 0);
    vec4 meta = texelFetch(uTreeMeta, tCoord, 0);

    vec3 d = pm.xyz - pos;
    float distSq = dot(d, d);
    float nodeSize = meta.x;

    if (nodeSize < 0.0 || nodeSize * nodeSize < uThetaSq * distSq) {
      if (distSq > 1e-8) {
        float epsPairSq = 0.5 * (selfEpsSq + meta.z);
        float denom = distSq + epsPairSq;
        float invR = inversesqrt(denom);
        float invR3 = pm.w * invR * invR * invR;
        acc += d * invR3;

        float rhoDenom = distSq + 16.0;
        float invRhoR = inversesqrt(rhoDenom);
        localDensity += pm.w * invRhoR * invRhoR * invRhoR;
      }
      node = int(meta.y + 0.5);
    } else {
      node = node + 1;
    }
  }

  acc *= uG;
  vel += uKickDt * acc;
  pos += uDriftDt * vel;

  if (uIsFirstStep == 1) {
    baselineDensity = max(localDensity, 1e-4);
    starburst = 0.0;
  } else {
    float ratio = localDensity / max(baselineDensity, 1e-4);
    bool canStarburst = (popType == 0 || popType == 4 || popType == 5);
    float excite = canStarburst
      ? 3.5 * uStepDt * pow(max(0.0, ratio - 1.18), 1.4)
      : 0.0;
    starburst = clamp(starburst * exp(-uStepDt / 0.14) + excite, 0.0, 2.5);
  }

  outPos = vec4(pos, starburst);
  outVel = vec4(vel, baselineDensity);
}
`;

export const GPGPU_APPROACH_FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uLocalInitPosTex;
uniform sampler2D uStaticColorTex;

uniform float uTimeGyr;
uniform float uTHandoff;
uniform mat3 uBasisMW;
uniform mat3 uBasisM31;
uniform vec3 uMwPos;
uniform vec3 uMwVel;
uniform vec3 uM31Pos;
uniform vec3 uM31Vel;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outVel;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec4 localData = texelFetch(uLocalInitPosTex, coord, 0);
  vec4 colData = texelFetch(uStaticColorTex, coord, 0);

  vec3 lp = localData.xyz;
  bool isM31 = localData.w > 0.5;
  float omega = colData.w;

  float dt = uTimeGyr - uTHandoff;
  float angle = omega * dt;
  float c = cos(angle);
  float s = sin(angle);

  vec3 rotPos = vec3(c * lp.x - s * lp.y, s * lp.x + c * lp.y, lp.z);
  vec3 rotVel = vec3(-omega * rotPos.y, omega * rotPos.x, 0.0);

  mat3 basis = isM31 ? uBasisM31 : uBasisMW;
  vec3 galPos = isM31 ? uM31Pos : uMwPos;
  vec3 galVel = isM31 ? uM31Vel : uMwVel;

  vec3 worldPos = basis * rotPos + galPos;
  vec3 worldVel = basis * rotVel + galVel;

  outPos = vec4(worldPos, 0.0);
  outVel = vec4(worldVel, 0.0);
}
`;

export const GPGPU_INIT_FRAG = /* glsl */ `
precision highp float;

uniform sampler2D uInitPosTex;
uniform sampler2D uInitVelTex;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outVel;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  outPos = texelFetch(uInitPosTex, coord, 0);
  outVel = texelFetch(uInitVelTex, coord, 0);
}
`;

export const GPGPU_RESTORE_FRAG = /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2DArray;

uniform sampler2DArray uPosArray;
uniform sampler2DArray uVelArray;
uniform int uSlice;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outVel;

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  outPos = texelFetch(uPosArray, ivec3(coord, uSlice), 0);
  outVel = texelFetch(uVelArray, ivec3(coord, uSlice), 0);
}
`;
