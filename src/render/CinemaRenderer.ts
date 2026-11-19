import * as THREE from 'three';
import { GpuParticleAdvect } from '../gpu/GpuParticleAdvect.ts';
import {
  VIS_SLICE_DUST_COUNT,
  VIS_SLICE_DUST_START,
  VIS_SLICE_NEBULA_COUNT,
  VIS_SLICE_NEBULA_START,
  VIS_SLICE_STARS_COUNT,
  VIS_SLICE_STARS_START,
} from '../gpu/visualParticles.ts';
import { Vec3 } from '../physics/constants.ts';
import { CORE_GLOW_FRAG, CORE_GLOW_VERT } from './shaders/coreGlowShader.ts';
import { DUST_FRAG, DUST_VERT } from './shaders/dustShader.ts';
import { NEBULA_FRAG, NEBULA_VERT } from './shaders/nebulaShader.ts';
import {
  ANAMORPHIC_BLUR_FRAG,
  ANAMORPHIC_THRESH_FRAG,
  BLOOM_DOWNSAMPLE_FRAG,
  BLOOM_PREFILTER_FRAG,
  BLOOM_UPSAMPLE_FRAG,
  COMPOSITE_FRAG,
  POST_QUAD_VERT,
} from './shaders/postProcessShaders.ts';
import { SKYBOX_FRAG, SKYBOX_VERT } from './shaders/skyboxShader.ts';
import { STAR_FRAG, STAR_VERT } from './shaders/starShader.ts';

function buildIndexPointsGeometry(startIdx: number, count: number): THREE.BufferGeometry {
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    positions[i * 3] = startIdx + i;
  }
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  return geom;
}

function makeHdrTarget(w: number, h: number): THREE.WebGLRenderTarget {
  return new THREE.WebGLRenderTarget(Math.max(1, w), Math.max(1, h), {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    wrapS: THREE.ClampToEdgeWrapping,
    wrapT: THREE.ClampToEdgeWrapping,
    generateMipmaps: false,
    depthBuffer: false,
    stencilBuffer: false,
  });
}

const BLOOM_MIP_COUNT = 6;

export class CinemaRenderer {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly advect: GpuParticleAdvect;

  private sceneRT: THREE.WebGLRenderTarget;
  private readonly bloomDownMips: THREE.WebGLRenderTarget[] = [];
  private readonly bloomUpMips: THREE.WebGLRenderTarget[] = [];
  private anamorphA: THREE.WebGLRenderTarget;
  private anamorphB: THREE.WebGLRenderTarget;

  private drawWidth = 1;
  private drawHeight = 1;

  // Scene pass objects
  private readonly galaxyScene: THREE.Scene;
  private readonly skyboxMat: THREE.RawShaderMaterial;
  private readonly nebulaMat: THREE.RawShaderMaterial;
  private readonly starMat: THREE.RawShaderMaterial;
  private readonly dustMat: THREE.RawShaderMaterial;
  private readonly coreGlowMat: THREE.RawShaderMaterial;

  private readonly nebulaPoints: THREE.Points;
  private readonly starPoints: THREE.Points;
  private readonly dustPoints: THREE.Points;
  private readonly coreGlowPoints: THREE.Points;
  private readonly corePosAttr: THREE.BufferAttribute;
  private readonly coreSizeAttr: THREE.BufferAttribute;
  private readonly coreIntAttr: THREE.BufferAttribute;

  // Post-process fullscreen quad
  private readonly postScene: THREE.Scene;
  private readonly postCamera: THREE.OrthographicCamera;
  private readonly postQuad: THREE.Mesh<THREE.PlaneGeometry, THREE.RawShaderMaterial>;
  private readonly prefilterMat: THREE.RawShaderMaterial;
  private readonly downsampleMat: THREE.RawShaderMaterial;
  private readonly upsampleMat: THREE.RawShaderMaterial;
  private readonly anamorphThreshMat: THREE.RawShaderMaterial;
  private readonly anamorphBlurMat: THREE.RawShaderMaterial;
  private readonly compositeMat: THREE.RawShaderMaterial;

  // Scratch vectors
  private readonly drawSize = new THREE.Vector2();
  private readonly camForward = new THREE.Vector3();
  private readonly camRight = new THREE.Vector3();
  private readonly camUp = new THREE.Vector3();
  private readonly mwCenterVec = new THREE.Vector3();
  private readonly m31CenterVec = new THREE.Vector3();

  constructor(renderer: THREE.WebGLRenderer, advect: GpuParticleAdvect) {
    this.renderer = renderer;
    this.advect = advect;

    renderer.autoClear = false;
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;

    renderer.getDrawingBufferSize(this.drawSize);
    this.drawWidth = Math.max(1, Math.floor(this.drawSize.x));
    this.drawHeight = Math.max(1, Math.floor(this.drawSize.y));

    this.sceneRT = makeHdrTarget(this.drawWidth, this.drawHeight);
    for (let i = 0; i < BLOOM_MIP_COUNT; i++) {
      const div = 1 << (i + 1);
      const mw = Math.max(1, Math.floor(this.drawWidth / div));
      const mh = Math.max(1, Math.floor(this.drawHeight / div));
      this.bloomDownMips.push(makeHdrTarget(mw, mh));
      if (i < BLOOM_MIP_COUNT - 1) {
        this.bloomUpMips.push(makeHdrTarget(mw, mh));
      }
    }
    this.anamorphA = makeHdrTarget(Math.floor(this.drawWidth / 4), Math.floor(this.drawHeight / 8));
    this.anamorphB = makeHdrTarget(Math.floor(this.drawWidth / 4), Math.floor(this.drawHeight / 8));

    // 1. Skybox pass
    this.galaxyScene = new THREE.Scene();
    const skyGeom = new THREE.PlaneGeometry(2, 2);
    this.skyboxMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: SKYBOX_VERT,
      fragmentShader: SKYBOX_FRAG,
      uniforms: {
        uCamRight: { value: new THREE.Vector3(1, 0, 0) },
        uCamUp: { value: new THREE.Vector3(0, 1, 0) },
        uCamForward: { value: new THREE.Vector3(0, 0, -1) },
        uTanHalfFov: { value: 0.38 },
        uAspect: { value: 16 / 9 },
      },
      depthTest: false,
      depthWrite: false,
    });
    const skyMesh = new THREE.Mesh(skyGeom, this.skyboxMat);
    skyMesh.frustumCulled = false;
    this.galaxyScene.add(skyMesh);

    // 2. Nebula & Gas Continuum pass
    const nebulaGeom = buildIndexPointsGeometry(VIS_SLICE_NEBULA_START, VIS_SLICE_NEBULA_COUNT);
    this.nebulaMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: NEBULA_VERT,
      fragmentShader: NEBULA_FRAG,
      uniforms: {
        uPosTex: { value: advect.posTexture },
        uVelTex: { value: advect.velTexture },
        uStaticAttrTex: { value: advect.staticAttrTexture },
        uStaticColorTex: { value: advect.staticColorTexture },
        uFocalLength: { value: 900.0 },
        uModelViewMatrix: { value: new THREE.Matrix4() },
        uProjectionMatrix: { value: new THREE.Matrix4() },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
    });
    this.nebulaPoints = new THREE.Points(nebulaGeom, this.nebulaMat);
    this.nebulaPoints.frustumCulled = false;

    // 3. Crisp Stars pass (reused for Far-side uSide=-1 and Near-side uSide=+1)
    const starGeom = buildIndexPointsGeometry(VIS_SLICE_STARS_START, VIS_SLICE_STARS_COUNT);
    this.starMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: STAR_VERT,
      fragmentShader: STAR_FRAG,
      uniforms: {
        uPosTex: { value: advect.posTexture },
        uStaticAttrTex: { value: advect.staticAttrTexture },
        uStaticColorTex: { value: advect.staticColorTexture },
        uMwCenter: { value: this.mwCenterVec },
        uM31Center: { value: this.m31CenterVec },
        uCamForward: { value: this.camForward },
        uSide: { value: -1.0 },
        uFocalLength: { value: 900.0 },
        uModelViewMatrix: { value: new THREE.Matrix4() },
        uProjectionMatrix: { value: new THREE.Matrix4() },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
    });
    this.starPoints = new THREE.Points(starGeom, this.starMat);
    this.starPoints.frustumCulled = false;

    // 4. Order-independent Cardelli-Clayton-Mathis Beer-Lambert Dust Absorption pass
    const dustGeom = buildIndexPointsGeometry(VIS_SLICE_DUST_START, VIS_SLICE_DUST_COUNT);
    this.dustMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: DUST_VERT,
      fragmentShader: DUST_FRAG,
      uniforms: {
        uPosTex: { value: advect.posTexture },
        uVelTex: { value: advect.velTexture },
        uStaticAttrTex: { value: advect.staticAttrTexture },
        uMwCenter: { value: this.mwCenterVec },
        uM31Center: { value: this.m31CenterVec },
        uCamForward: { value: this.camForward },
        uFocalLength: { value: 900.0 },
        uModelViewMatrix: { value: new THREE.Matrix4() },
        uProjectionMatrix: { value: new THREE.Matrix4() },
      },
      transparent: true,
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.ZeroFactor,
      blendDst: THREE.OneMinusSrcColorFactor,
      blendEquationAlpha: THREE.AddEquation,
      blendSrcAlpha: THREE.ZeroFactor,
      blendDstAlpha: THREE.OneFactor,
      depthTest: false,
      depthWrite: false,
    });
    this.dustPoints = new THREE.Points(dustGeom, this.dustMat);
    this.dustPoints.frustumCulled = false;

    // 5. Dual Galactic Core Glows
    const coreGeom = new THREE.BufferGeometry();
    this.corePosAttr = new THREE.BufferAttribute(new Float32Array(6), 3);
    this.coreSizeAttr = new THREE.BufferAttribute(new Float32Array([18.0, 26.0]), 1);
    this.coreIntAttr = new THREE.BufferAttribute(new Float32Array([0.85, 1.15]), 1);
    coreGeom.setAttribute('position', this.corePosAttr);
    coreGeom.setAttribute('aSizeKpc', this.coreSizeAttr);
    coreGeom.setAttribute('aIntensity', this.coreIntAttr);

    this.coreGlowMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: CORE_GLOW_VERT,
      fragmentShader: CORE_GLOW_FRAG,
      uniforms: {
        uFocalLength: { value: 900.0 },
        uModelViewMatrix: { value: new THREE.Matrix4() },
        uProjectionMatrix: { value: new THREE.Matrix4() },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthTest: false,
      depthWrite: false,
    });
    this.coreGlowPoints = new THREE.Points(coreGeom, this.coreGlowMat);
    this.coreGlowPoints.frustumCulled = false;

    // 6. Post-process passes
    this.postScene = new THREE.Scene();
    this.postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    this.prefilterMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: POST_QUAD_VERT,
      fragmentShader: BLOOM_PREFILTER_FRAG,
      uniforms: {
        uInputTex: { value: null },
        uTexelSize: { value: new THREE.Vector2() },
        uThreshold: { value: 0.62 },
        uKnee: { value: 0.55 },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.downsampleMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: POST_QUAD_VERT,
      fragmentShader: BLOOM_DOWNSAMPLE_FRAG,
      uniforms: {
        uInputTex: { value: null },
        uTexelSize: { value: new THREE.Vector2() },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.upsampleMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: POST_QUAD_VERT,
      fragmentShader: BLOOM_UPSAMPLE_FRAG,
      uniforms: {
        uHigherTex: { value: null },
        uLowerTex: { value: null },
        uLowerTexelSize: { value: new THREE.Vector2() },
        uRadius: { value: 1.15 },
        uWeight: { value: 0.72 },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.anamorphThreshMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: POST_QUAD_VERT,
      fragmentShader: ANAMORPHIC_THRESH_FRAG,
      uniforms: {
        uInputTex: { value: null },
        uThreshold: { value: 3.0 },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.anamorphBlurMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: POST_QUAD_VERT,
      fragmentShader: ANAMORPHIC_BLUR_FRAG,
      uniforms: {
        uInputTex: { value: null },
        uStepX: { value: 0.004 },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.compositeMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: POST_QUAD_VERT,
      fragmentShader: COMPOSITE_FRAG,
      uniforms: {
        uSceneTex: { value: this.sceneRT.texture },
        uBloomTex: { value: this.bloomUpMips[0].texture },
        uAnamorphicTex: { value: this.anamorphA.texture },
        uExposure: { value: 1.26 },
        uBloomStrength: { value: 0.52 },
        uAnamorphicStrength: { value: 0.18 },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.postQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compositeMat);
    this.postQuad.frustumCulled = false;
    this.postScene.add(this.postQuad);
  }

  public resize(): void {
    this.renderer.getDrawingBufferSize(this.drawSize);
    const w = Math.max(1, Math.floor(this.drawSize.x));
    const h = Math.max(1, Math.floor(this.drawSize.y));
    if (w === this.drawWidth && h === this.drawHeight) return;

    this.drawWidth = w;
    this.drawHeight = h;

    this.sceneRT.setSize(w, h);
    for (let i = 0; i < BLOOM_MIP_COUNT; i++) {
      const div = 1 << (i + 1);
      const mw = Math.max(1, Math.floor(w / div));
      const mh = Math.max(1, Math.floor(h / div));
      this.bloomDownMips[i].setSize(mw, mh);
      if (i < BLOOM_MIP_COUNT - 1) {
        this.bloomUpMips[i].setSize(mw, mh);
      }
    }
    this.anamorphA.setSize(Math.max(1, Math.floor(w / 4)), Math.max(1, Math.floor(h / 8)));
    this.anamorphB.setSize(Math.max(1, Math.floor(w / 4)), Math.max(1, Math.floor(h / 8)));
  }

  private renderPassObject(
    obj: THREE.Object3D,
    camera: THREE.Camera,
  ): void {
    // Render single object cleanly inside a temporary group or toggle visibility
    // Using a dedicated helper scene avoided allocation, here we add/keep in galaxyScene and toggle visibility
    obj.visible = true;
    this.renderer.render(this.galaxyScene, camera);
    obj.visible = false;
  }

  public render(
    camera: THREE.PerspectiveCamera,
    mwCenter: Vec3,
    m31Center: Vec3,
    sepKpc: number,
  ): void {
    this.resize();
    camera.updateMatrixWorld();

    // Physical-pixel focal length for consistent HiDPI point scaling
    const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
    const focalLengthPx = (0.5 * this.drawHeight) / tanHalfFov;

    camera.matrixWorld.extractBasis(this.camRight, this.camUp, this.camForward);
    this.camForward.negate(); // camera looks along -Z in world basis

    this.mwCenterVec.set(mwCenter[0], mwCenter[1], mwCenter[2]);
    this.m31CenterVec.set(m31Center[0], m31Center[1], m31Center[2]);

    const posTex = this.advect.posTexture;
    const velTex = this.advect.velTexture;

    // ---------------------------------------------------------------------
    // Pass 1..4 into HDR sceneRT
    // ---------------------------------------------------------------------
    this.renderer.setRenderTarget(this.sceneRT);
    this.renderer.clear(true, false, false);

    // Ensure galaxyScene contains our ordered renderables lazily once
    if (this.galaxyScene.children.length === 1) {
      this.galaxyScene.add(
        this.nebulaPoints,
        this.starPoints,
        this.dustPoints,
        this.coreGlowPoints,
      );
      this.nebulaPoints.visible = false;
      this.starPoints.visible = false;
      this.dustPoints.visible = false;
      this.coreGlowPoints.visible = false;
    }

    const skyMesh = this.galaxyScene.children[0];

    // Pass 1: Procedural Hubble Deep Field Skybox
    (this.skyboxMat.uniforms.uCamRight.value as THREE.Vector3).copy(this.camRight);
    (this.skyboxMat.uniforms.uCamUp.value as THREE.Vector3).copy(this.camUp);
    (this.skyboxMat.uniforms.uCamForward.value as THREE.Vector3).copy(this.camForward);
    this.skyboxMat.uniforms.uTanHalfFov.value = tanHalfFov;
    this.skyboxMat.uniforms.uAspect.value = this.drawWidth / Math.max(1, this.drawHeight);
    skyMesh.visible = true;
    this.renderer.render(this.galaxyScene, camera);
    skyMesh.visible = false;

    // Pass 2a: Volumetric Gas Continuum & H-II Emission Nebulae
    this.nebulaMat.uniforms.uPosTex.value = posTex;
    this.nebulaMat.uniforms.uVelTex.value = velTex;
    this.nebulaMat.uniforms.uFocalLength.value = focalLengthPx;
    (this.nebulaMat.uniforms.uModelViewMatrix.value as THREE.Matrix4).copy(camera.matrixWorldInverse);
    (this.nebulaMat.uniforms.uProjectionMatrix.value as THREE.Matrix4).copy(camera.projectionMatrix);
    this.renderPassObject(this.nebulaPoints, camera);

    // Pass 2b: Far-side Crisp Stars (behind dust lanes)
    this.starMat.uniforms.uPosTex.value = posTex;
    this.starMat.uniforms.uSide.value = -1.0;
    this.starMat.uniforms.uFocalLength.value = focalLengthPx;
    (this.starMat.uniforms.uModelViewMatrix.value as THREE.Matrix4).copy(camera.matrixWorldInverse);
    (this.starMat.uniforms.uProjectionMatrix.value as THREE.Matrix4).copy(camera.projectionMatrix);
    this.renderPassObject(this.starPoints, camera);

    // Pass 3: Order-independent Beer-Lambert Dark Molecular Dust Extinction
    this.dustMat.uniforms.uPosTex.value = posTex;
    this.dustMat.uniforms.uVelTex.value = velTex;
    this.dustMat.uniforms.uFocalLength.value = focalLengthPx;
    (this.dustMat.uniforms.uModelViewMatrix.value as THREE.Matrix4).copy(camera.matrixWorldInverse);
    (this.dustMat.uniforms.uProjectionMatrix.value as THREE.Matrix4).copy(camera.projectionMatrix);
    this.renderPassObject(this.dustPoints, camera);

    // Pass 4a: Near-side Crisp Stars (in front of dust lanes)
    this.starMat.uniforms.uSide.value = 1.0;
    this.renderPassObject(this.starPoints, camera);

    // Pass 4b: Galactic Bulge Core Glows (blend cleanly into single Milkomeda nucleus)
    const mergeProximity = Math.max(0, 1.0 - sepKpc / 18.0);
    const posArr = this.corePosAttr.array as Float32Array;
    posArr[0] = mwCenter[0]; posArr[1] = mwCenter[1]; posArr[2] = mwCenter[2];
    posArr[3] = m31Center[0]; posArr[4] = m31Center[1]; posArr[5] = m31Center[2];
    this.corePosAttr.needsUpdate = true;

    const intArr = this.coreIntAttr.array as Float32Array;
    intArr[0] = 0.24 * (1.0 - 0.35 * mergeProximity);
    intArr[1] = 0.32 * (1.0 - 0.15 * mergeProximity);
    this.coreIntAttr.needsUpdate = true;

    const sizeArr = this.coreSizeAttr.array as Float32Array;
    sizeArr[0] = 11.0;
    sizeArr[1] = 15.0 + 3.0 * mergeProximity;
    this.coreSizeAttr.needsUpdate = true;

    this.coreGlowMat.uniforms.uFocalLength.value = focalLengthPx;
    (this.coreGlowMat.uniforms.uModelViewMatrix.value as THREE.Matrix4).copy(camera.matrixWorldInverse);
    (this.coreGlowMat.uniforms.uProjectionMatrix.value as THREE.Matrix4).copy(camera.projectionMatrix);
    this.renderPassObject(this.coreGlowPoints, camera);

    // ---------------------------------------------------------------------
    // Pass 5: 6-Mip Karis HDR Bloom Pyramid + Anamorphic Streak
    // ---------------------------------------------------------------------
    this.prefilterMat.uniforms.uInputTex.value = this.sceneRT.texture;
    (this.prefilterMat.uniforms.uTexelSize.value as THREE.Vector2).set(
      1.0 / this.drawWidth,
      1.0 / this.drawHeight,
    );
    this.postQuad.material = this.prefilterMat;
    this.renderer.setRenderTarget(this.bloomDownMips[0]);
    this.renderer.render(this.postScene, this.postCamera);

    this.postQuad.material = this.downsampleMat;
    for (let i = 1; i < BLOOM_MIP_COUNT; i++) {
      const prev = this.bloomDownMips[i - 1];
      const dst = this.bloomDownMips[i];
      this.downsampleMat.uniforms.uInputTex.value = prev.texture;
      (this.downsampleMat.uniforms.uTexelSize.value as THREE.Vector2).set(
        1.0 / prev.width,
        1.0 / prev.height,
      );
      this.renderer.setRenderTarget(dst);
      this.renderer.render(this.postScene, this.postCamera);
    }

    // Progressive tent upsample from mip[5] -> up[4] -> ... -> up[0]
    this.postQuad.material = this.upsampleMat;
    let lowerTex = this.bloomDownMips[BLOOM_MIP_COUNT - 1].texture;
    let lowerW = this.bloomDownMips[BLOOM_MIP_COUNT - 1].width;
    let lowerH = this.bloomDownMips[BLOOM_MIP_COUNT - 1].height;

    for (let i = BLOOM_MIP_COUNT - 2; i >= 0; i--) {
      const higher = this.bloomDownMips[i];
      const dst = this.bloomUpMips[i];
      this.upsampleMat.uniforms.uHigherTex.value = higher.texture;
      this.upsampleMat.uniforms.uLowerTex.value = lowerTex;
      (this.upsampleMat.uniforms.uLowerTexelSize.value as THREE.Vector2).set(
        1.0 / lowerW,
        1.0 / lowerH,
      );
      this.upsampleMat.uniforms.uWeight.value = i === 0 ? 0.55 : 0.72;
      this.renderer.setRenderTarget(dst);
      this.renderer.render(this.postScene, this.postCamera);
      lowerTex = dst.texture;
      lowerW = dst.width;
      lowerH = dst.height;
    }

    // Subtle horizontal anamorphic streak from extreme HDR highlights
    this.anamorphThreshMat.uniforms.uInputTex.value = this.bloomDownMips[0].texture;
    this.postQuad.material = this.anamorphThreshMat;
    this.renderer.setRenderTarget(this.anamorphA);
    this.renderer.render(this.postScene, this.postCamera);

    this.postQuad.material = this.anamorphBlurMat;
    this.anamorphBlurMat.uniforms.uInputTex.value = this.anamorphA.texture;
    this.anamorphBlurMat.uniforms.uStepX.value = 1.4 / Math.max(1, this.anamorphA.width);
    this.renderer.setRenderTarget(this.anamorphB);
    this.renderer.render(this.postScene, this.postCamera);

    this.anamorphBlurMat.uniforms.uInputTex.value = this.anamorphB.texture;
    this.anamorphBlurMat.uniforms.uStepX.value = 2.8 / Math.max(1, this.anamorphB.width);
    this.renderer.setRenderTarget(this.anamorphA);
    this.renderer.render(this.postScene, this.postCamera);

    // ---------------------------------------------------------------------
    // Pass 6: Final Hollywood AgX Composite -> Screen
    // ---------------------------------------------------------------------
    this.compositeMat.uniforms.uSceneTex.value = this.sceneRT.texture;
    this.compositeMat.uniforms.uBloomTex.value = this.bloomUpMips[0].texture;
    this.compositeMat.uniforms.uAnamorphicTex.value = this.anamorphA.texture;
    this.postQuad.material = this.compositeMat;
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.postScene, this.postCamera);
  }

  public dispose(): void {
    this.sceneRT.dispose();
    for (const m of this.bloomDownMips) m.dispose();
    for (const m of this.bloomUpMips) m.dispose();
    this.anamorphA.dispose();
    this.anamorphB.dispose();

    this.skyboxMat.dispose();
    this.nebulaMat.dispose();
    this.starMat.dispose();
    this.dustMat.dispose();
    this.coreGlowMat.dispose();
    this.prefilterMat.dispose();
    this.downsampleMat.dispose();
    this.upsampleMat.dispose();
    this.anamorphThreshMat.dispose();
    this.anamorphBlurMat.dispose();
    this.compositeMat.dispose();

    this.nebulaPoints.geometry.dispose();
    this.starPoints.geometry.dispose();
    this.dustPoints.geometry.dispose();
    this.coreGlowPoints.geometry.dispose();
    this.postQuad.geometry.dispose();
    (this.galaxyScene.children[0] as THREE.Mesh).geometry.dispose();
  }
}
