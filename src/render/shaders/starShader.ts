export const STAR_VERT = /* glsl */ `
precision highp float;
precision highp int;

in vec3 position;

uniform sampler2D uPosTex;
uniform sampler2D uStaticAttrTex;
uniform sampler2D uStaticColorTex;

uniform vec3 uMwCenter;
uniform vec3 uM31Center;
uniform vec3 uCamForward;
uniform float uSide; // -1.0 = far side behind dust, +1.0 = near side in front of dust
uniform float uFocalLength;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

out vec3 vColor;
out float vIntensity;
out float vFlare;
out float vAngle;

void main() {
  int idx = int(position.x + 0.5);
  ivec2 coord = ivec2(idx & 511, idx >> 9);

  vec4 posData = texelFetch(uPosTex, coord, 0);
  vec4 attr = texelFetch(uStaticAttrTex, coord, 0);
  vec4 colData = texelFetch(uStaticColorTex, coord, 0);

  int packedType = int(attr.x + 0.5);
  int galId = packedType / 10;
  int popType = packedType - galId * 10;

  vec3 hostCenter = (galId == 0) ? uMwCenter : uM31Center;
  float relDepth = dot(posData.xyz - hostCenter, uCamForward);

  // Split stars into far-side (>= 0) and near-side (< 0) around host dust plane
  if ((uSide < 0.0 && relDepth < 0.0) || (uSide > 0.0 && relDepth >= 0.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    vIntensity = 0.0;
    return;
  }

  vec4 mvPos = uModelViewMatrix * vec4(posData.xyz, 1.0);
  float dist = -mvPos.z;
  if (dist <= 0.2) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    vIntensity = 0.0;
    return;
  }

  gl_Position = uProjectionMatrix * mvPos;

  float baseSizeKpc = attr.y;
  float lum = attr.z;
  float starburst = posData.w;

  float rawPx = (baseSizeKpc * uFocalLength) / dist;
  float clampedPx = clamp(rawPx, 1.35, 14.0);
  gl_PointSize = clampedPx;

  // Sub-pixel energy conservation eliminates twinkling/shimmering during motion
  float subRatio = clamp(rawPx / clampedPx, 0.14, 1.0);
  float energy = pow(subRatio, 1.45);

  float nearFade = smoothstep(0.6, 4.0, dist);
  float burstBoost = (popType == 0) ? (1.0 + 1.55 * starburst) : 1.0;

  vec3 col = colData.rgb;
  if (popType == 0 && starburst > 0.05) {
    col = mix(col, vec3(0.70, 0.88, 1.0), clamp(starburst * 0.45, 0.0, 0.65));
  }

  vColor = col;
  vIntensity = clamp(lum * burstBoost * energy * nearFade * 0.44, 0.0, 8.0);
  vFlare = (lum > 1.65 && clampedPx > 3.0) ? clamp((lum - 1.55) * 0.42, 0.0, 1.15) : 0.0;
  vAngle = attr.w * 1.570796;
}
`;

export const STAR_FRAG = /* glsl */ `
precision highp float;

in vec3 vColor;
in float vIntensity;
in float vFlare;
in float vAngle;

out vec4 fragColor;

void main() {
  if (vIntensity <= 0.0005) discard;

  vec2 uv = gl_PointCoord - 0.5;
  float r = length(uv);
  if (r > 0.5) discard;

  float r2 = dot(uv, uv);
  // Crisp Airy-disk core + delicate optical corona
  float airy = exp(-r2 * 30.0) + 0.24 * exp(-r2 * 8.5);
  float edgeMask = smoothstep(0.5, 0.34, r);

  float flare = 0.0;
  if (vFlare > 0.01) {
    float ca = cos(vAngle);
    float sa = sin(vAngle);
    vec2 fuv = vec2(ca * uv.x + sa * uv.y, -sa * uv.x + ca * uv.y);
    float s1 = exp(-abs(fuv.x) * 32.0) * exp(-abs(fuv.y) * 7.0);
    float s2 = exp(-abs(fuv.y) * 32.0) * exp(-abs(fuv.x) * 7.0);
    flare = (s1 + s2) * vFlare * 0.55;
  }

  float shape = (airy + flare) * edgeMask;
  fragColor = vec4(vColor * (vIntensity * shape), 1.0);
}
`;
