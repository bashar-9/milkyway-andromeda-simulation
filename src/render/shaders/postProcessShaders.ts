export const POST_QUAD_VERT = /* glsl */ `
precision highp float;

in vec3 position;
in vec2 uv;
out vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// Soft-knee HDR prefilter + Karis-weighted first downsample (eliminates star flicker)
export const BLOOM_PREFILTER_FRAG = /* glsl */ `
precision highp float;

in vec2 vUv;
uniform sampler2D uInputTex;
uniform vec2 uTexelSize;
uniform float uThreshold;
uniform float uKnee;

out vec4 fragColor;

float luminance(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 softThreshold(vec3 c) {
  float br = max(c.r, max(c.g, c.b));
  float rq = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  rq = (rq * rq) / max(4.0 * uKnee, 1e-5);
  float weight = max(rq, br - uThreshold) / max(br, 1e-5);
  return c * weight;
}

vec3 karisSample(vec2 uv) {
  vec3 c = texture(uInputTex, uv).rgb;
  return softThreshold(min(c, vec3(24.0)));
}

void main() {
  vec2 d = uTexelSize;

  vec3 a = karisSample(vUv + vec2(-d.x,  d.y));
  vec3 b = karisSample(vUv + vec2( d.x,  d.y));
  vec3 c = karisSample(vUv + vec2(-d.x, -d.y));
  vec3 e = karisSample(vUv + vec2( d.x, -d.y));
  vec3 m = karisSample(vUv);

  float wa = 1.0 / (1.0 + luminance(a));
  float wb = 1.0 / (1.0 + luminance(b));
  float wc = 1.0 / (1.0 + luminance(c));
  float we = 1.0 / (1.0 + luminance(e));
  float wm = 1.0 / (1.0 + luminance(m));

  vec3 sum = a * wa + b * wb + c * wc + e * we + m * wm * 2.0;
  float wSum = wa + wb + wc + we + wm * 2.0;
  fragColor = vec4(sum / max(wSum, 1e-5), 1.0);
}
`;

// 13-tap Jimenez progressive downsample
export const BLOOM_DOWNSAMPLE_FRAG = /* glsl */ `
precision highp float;

in vec2 vUv;
uniform sampler2D uInputTex;
uniform vec2 uTexelSize;

out vec4 fragColor;

void main() {
  vec2 d = uTexelSize;

  vec3 a = texture(uInputTex, vUv + vec2(-2.0 * d.x,  2.0 * d.y)).rgb;
  vec3 b = texture(uInputTex, vUv + vec2( 0.0,        2.0 * d.y)).rgb;
  vec3 c = texture(uInputTex, vUv + vec2( 2.0 * d.x,  2.0 * d.y)).rgb;

  vec3 e = texture(uInputTex, vUv + vec2(-2.0 * d.x,  0.0)).rgb;
  vec3 f = texture(uInputTex, vUv).rgb;
  vec3 g = texture(uInputTex, vUv + vec2( 2.0 * d.x,  0.0)).rgb;

  vec3 h = texture(uInputTex, vUv + vec2(-2.0 * d.x, -2.0 * d.y)).rgb;
  vec3 i = texture(uInputTex, vUv + vec2( 0.0,       -2.0 * d.y)).rgb;
  vec3 j = texture(uInputTex, vUv + vec2( 2.0 * d.x, -2.0 * d.y)).rgb;

  vec3 k = texture(uInputTex, vUv + vec2(-d.x,  d.y)).rgb;
  vec3 l = texture(uInputTex, vUv + vec2( d.x,  d.y)).rgb;
  vec3 m = texture(uInputTex, vUv + vec2(-d.x, -d.y)).rgb;
  vec3 n = texture(uInputTex, vUv + vec2( d.x, -d.y)).rgb;

  vec3 res = f * 0.125;
  res += (a + c + h + j) * 0.03125;
  res += (b + e + g + i) * 0.0625;
  res += (k + l + m + n) * 0.125;

  fragColor = vec4(res, 1.0);
}
`;

// 9-tap tent filter upsample blended with current mip
export const BLOOM_UPSAMPLE_FRAG = /* glsl */ `
precision highp float;

in vec2 vUv;
uniform sampler2D uHigherTex;
uniform sampler2D uLowerTex;
uniform vec2 uLowerTexelSize;
uniform float uRadius;
uniform float uWeight;

out vec4 fragColor;

void main() {
  vec2 d = uLowerTexelSize * uRadius;

  vec3 s = vec3(0.0);
  s += texture(uLowerTex, vUv + vec2(-d.x,  d.y)).rgb * 1.0;
  s += texture(uLowerTex, vUv + vec2( 0.0,  d.y)).rgb * 2.0;
  s += texture(uLowerTex, vUv + vec2( d.x,  d.y)).rgb * 1.0;
  s += texture(uLowerTex, vUv + vec2(-d.x,  0.0)).rgb * 2.0;
  s += texture(uLowerTex, vUv).rgb * 4.0;
  s += texture(uLowerTex, vUv + vec2( d.x,  0.0)).rgb * 2.0;
  s += texture(uLowerTex, vUv + vec2(-d.x, -d.y)).rgb * 1.0;
  s += texture(uLowerTex, vUv + vec2( 0.0, -d.y)).rgb * 2.0;
  s += texture(uLowerTex, vUv + vec2( d.x, -d.y)).rgb * 1.0;
  s *= (1.0 / 16.0);

  vec3 hi = texture(uHigherTex, vUv).rgb;
  fragColor = vec4(hi + s * uWeight, 1.0);
}
`;

// Anamorphic horizontal streak prefilter (only triggers on intense HDR highlights)
export const ANAMORPHIC_THRESH_FRAG = /* glsl */ `
precision highp float;

in vec2 vUv;
uniform sampler2D uInputTex;
uniform float uThreshold;

out vec4 fragColor;

void main() {
  vec3 c = texture(uInputTex, vUv).rgb;
  float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float excess = max(0.0, luma - uThreshold);
  vec3 res = c * (excess / max(luma, 1e-5));
  fragColor = vec4(min(res, vec3(8.0)), 1.0);
}
`;

// Wide separable horizontal streak blur
export const ANAMORPHIC_BLUR_FRAG = /* glsl */ `
precision highp float;

in vec2 vUv;
uniform sampler2D uInputTex;
uniform float uStepX;

out vec4 fragColor;

void main() {
  const float w0 = 0.145;
  const float w1 = 0.132;
  const float w2 = 0.106;
  const float w3 = 0.077;
  const float w4 = 0.052;
  const float w5 = 0.032;
  const float w6 = 0.018;
  const float w7 = 0.009;

  vec3 sum = texture(uInputTex, vUv).rgb * w0;
  sum += (texture(uInputTex, vUv + vec2( 1.0 * uStepX, 0.0)).rgb + texture(uInputTex, vUv - vec2( 1.0 * uStepX, 0.0)).rgb) * w1;
  sum += (texture(uInputTex, vUv + vec2( 2.0 * uStepX, 0.0)).rgb + texture(uInputTex, vUv - vec2( 2.0 * uStepX, 0.0)).rgb) * w2;
  sum += (texture(uInputTex, vUv + vec2( 3.5 * uStepX, 0.0)).rgb + texture(uInputTex, vUv - vec2( 3.5 * uStepX, 0.0)).rgb) * w3;
  sum += (texture(uInputTex, vUv + vec2( 5.5 * uStepX, 0.0)).rgb + texture(uInputTex, vUv - vec2( 5.5 * uStepX, 0.0)).rgb) * w4;
  sum += (texture(uInputTex, vUv + vec2( 8.0 * uStepX, 0.0)).rgb + texture(uInputTex, vUv - vec2( 8.0 * uStepX, 0.0)).rgb) * w5;
  sum += (texture(uInputTex, vUv + vec2(11.0 * uStepX, 0.0)).rgb + texture(uInputTex, vUv - vec2(11.0 * uStepX, 0.0)).rgb) * w6;
  sum += (texture(uInputTex, vUv + vec2(15.0 * uStepX, 0.0)).rgb + texture(uInputTex, vUv - vec2(15.0 * uStepX, 0.0)).rgb) * w7;

  fragColor = vec4(sum, 1.0);
}
`;

// Final Hollywood Cinema Composite:
// Scene + Bloom + Anamorphic + Edge CA + Troy Sobotka AgX Tone Map + Vignette + sRGB + Triangular Dither
export const COMPOSITE_FRAG = /* glsl */ `
precision highp float;

in vec2 vUv;
uniform sampler2D uSceneTex;
uniform sampler2D uBloomTex;
uniform sampler2D uAnamorphicTex;
uniform float uExposure;
uniform float uBloomStrength;
uniform float uAnamorphicStrength;

out vec4 fragColor;

// Troy Sobotka AgX default contrast approximation (sigmoid polynomial)
vec3 agxDefaultContrastApprox(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  return + 15.5     * x4 * x2
         - 40.14    * x4 * x
         + 31.96    * x4
         - 6.868    * x2 * x
         + 0.4298   * x2
         + 0.1191   * x
         - 0.00232;
}

vec3 agxToneMap(vec3 val) {
  const mat3 agxMat = mat3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104
  );
  const mat3 agxMatInv = mat3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
   -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
   -0.0990297440797205, -0.0989611768448433, 1.15107367264116
  );

  const float minEv = -12.47393;
  const float maxEv = 4.026069;

  val = max(val, vec3(1e-6));
  val = agxMat * val;
  val = clamp(log2(val), minEv, maxEv);
  val = (val - minEv) / (maxEv - minEv);
  val = agxDefaultContrastApprox(val);
  val = agxMatInv * val;
  return clamp(val, 0.0, 1.0);
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec2 centered = vUv - 0.5;
  float r2 = dot(centered, centered);

  // Subtle edge optical chromatic aberration
  vec2 caOffset = centered * (0.00075 * r2 * 4.0);
  float rChan = texture(uSceneTex, vUv + caOffset).r;
  float gChan = texture(uSceneTex, vUv).g;
  float bChan = texture(uSceneTex, vUv - caOffset).b;
  vec3 scene = vec3(rChan, gChan, bChan);

  vec3 bloom = texture(uBloomTex, vUv).rgb;
  vec3 anamorphic = texture(uAnamorphicTex, vUv).rgb * vec3(0.76, 0.88, 1.0);

  vec3 hdr = (scene + bloom * uBloomStrength + anamorphic * uAnamorphicStrength) * uExposure;

  vec3 mapped = agxToneMap(hdr);
  // AgX Punchy contrast toe: eliminates shadow pedestal so space is velvety obsidian
  mapped = max(vec3(0.0), (mapped - vec3(0.0045)) / 0.9955);
  mapped = pow(mapped, vec3(1.07));

  // Subtle optical vignette
  float vig = 1.0 - 0.18 * smoothstep(0.18, 0.85, r2 * 2.2);
  mapped *= vig;

  // Linear -> sRGB encoding
  vec3 srgb = mix(
    mapped * 12.92,
    1.055 * pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055,
    step(vec3(0.0031308), mapped)
  );

  // Triangular-PDF dithering (+/- 0.5 / 255) eliminates 8-bit deep-space banding
  float d1 = hash12(gl_FragCoord.xy);
  float d2 = hash12(gl_FragCoord.xy + vec2(37.0, 17.0));
  srgb += (d1 + d2 - 1.0) * (0.65 / 255.0);

  fragColor = vec4(clamp(srgb, 0.0, 1.0), 1.0);
}
`;
