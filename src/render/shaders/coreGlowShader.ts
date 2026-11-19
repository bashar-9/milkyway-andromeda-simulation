export const CORE_GLOW_VERT = /* glsl */ `
precision highp float;

in vec3 position;
in float aSizeKpc;
in float aIntensity;

uniform float uFocalLength;
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

out float vIntensity;

void main() {
  vec4 mvPos = uModelViewMatrix * vec4(position, 1.0);
  float dist = -mvPos.z;
  if (dist <= 0.3) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    vIntensity = 0.0;
    return;
  }

  gl_Position = uProjectionMatrix * mvPos;
  gl_PointSize = clamp((aSizeKpc * uFocalLength) / dist, 4.0, 420.0);
  vIntensity = aIntensity * smoothstep(1.5, 9.0, dist);
}
`;

export const CORE_GLOW_FRAG = /* glsl */ `
precision highp float;

in float vIntensity;
out vec4 fragColor;

void main() {
  if (vIntensity <= 0.001) discard;

  vec2 uv = gl_PointCoord - 0.5;
  float r = length(uv);
  if (r > 0.5) discard;

  float r2 = dot(uv, uv);
  // Multi-scale bulge core profile: incandescent center + soft warm outer halo
  float inner = exp(-r2 * 110.0) * 1.15;
  float mid   = exp(-r2 * 34.0) * 0.42;
  float outer = exp(-r2 * 14.0) * 0.14;
  float mask  = smoothstep(0.49, 0.22, r);

  vec3 innerCol = vec3(1.0, 0.92, 0.78);
  vec3 outerCol = vec3(1.0, 0.72, 0.38);
  vec3 col = innerCol * inner + outerCol * (mid + outer);

  fragColor = vec4(col * (vIntensity * mask), 1.0);
}
`;
