import * as THREE from 'three';
import {
  BASIS_M31_VIEW,
  BASIS_MW_VIEW,
  CHECKPOINT_SLICE_COUNT,
  G,
  Mat3,
  THETA_SQ,
  TREE_TEX_HEIGHT,
  TREE_TEX_WIDTH,
  T_HANDOFF,
  VIS_TEX_H,
  VIS_TEX_W,
  Vec3,
} from '../physics/constants.ts';
import {
  GPGPU_ADVECT_FRAG,
  GPGPU_APPROACH_FRAG,
  GPGPU_INIT_FRAG,
  GPGPU_QUAD_VERT,
  GPGPU_RESTORE_FRAG,
} from './gpgpuShaders.ts';
import { VisualParticlesData } from './visualParticles.ts';

function toThreeMat3(m: Mat3): THREE.Matrix3 {
  // Mat3 in constants.ts is row-major; THREE.Matrix3.set takes row-major arguments
  return new THREE.Matrix3().set(
    m[0], m[1], m[2],
    m[3], m[4], m[5],
    m[6], m[7], m[8],
  );
}

function makeFloatDataTexture(
  data: Float32Array,
  width: number,
  height: number,
): THREE.DataTexture {
  const tex = new THREE.DataTexture(
    data,
    width,
    height,
    THREE.RGBAFormat,
    THREE.FloatType,
  );
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Failed to create WebGL2 shader');
  gl.shaderSource(shader, `#version 300 es\n${source}`);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`GPGPU shader compile error: ${log}`);
  }
  return shader;
}

export class GpuParticleAdvect {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly gl: WebGL2RenderingContext;

  private readonly rtA: THREE.WebGLRenderTarget;
  private readonly rtB: THREE.WebGLRenderTarget;
  private activeIdx = 0;

  private readonly treePosMassData = new Float32Array(TREE_TEX_WIDTH * TREE_TEX_HEIGHT * 4);
  private readonly treeMetaData = new Float32Array(TREE_TEX_WIDTH * TREE_TEX_HEIGHT * 4);
  private readonly treePosMassTex: THREE.DataTexture;
  private readonly treeMetaTex: THREE.DataTexture;
  private currentNodeCount = 0;

  public readonly staticAttrTexture: THREE.DataTexture;
  public readonly staticColorTexture: THREE.DataTexture;
  private readonly initPosTex: THREE.DataTexture;
  private readonly initVelTex: THREE.DataTexture;
  private readonly localInitPosTex: THREE.DataTexture;

  private readonly quadScene: THREE.Scene;
  private readonly quadCamera: THREE.OrthographicCamera;
  private readonly quadMesh: THREE.Mesh<THREE.PlaneGeometry, THREE.RawShaderMaterial>;

  private readonly initMat: THREE.RawShaderMaterial;
  private readonly advectMat: THREE.RawShaderMaterial;
  private readonly approachMat: THREE.RawShaderMaterial;

  // Native WebGL2 3D texture arrays for zero-copy VRAM checkpoints
  private readonly posArrayTex: WebGLTexture;
  private readonly velArrayTex: WebGLTexture;
  private readonly restoreProgram: WebGLProgram;
  private readonly restoreVao: WebGLVertexArrayObject;
  private readonly restoreVbo: WebGLBuffer;
  private readonly restoreLocPosArray: WebGLUniformLocation | null;
  private readonly restoreLocVelArray: WebGLUniformLocation | null;
  private readonly restoreLocSlice: WebGLUniformLocation | null;

  constructor(renderer: THREE.WebGLRenderer, data: VisualParticlesData) {
    this.renderer = renderer;
    const gl = renderer.getContext() as WebGL2RenderingContext;
    this.gl = gl;

    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('WebGL2 EXT_color_buffer_float extension is not supported on this GPU/browser');
    }

    const rtOpts: THREE.RenderTargetOptions = {
      count: 2,
      type: THREE.FloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      wrapS: THREE.ClampToEdgeWrapping,
      wrapT: THREE.ClampToEdgeWrapping,
      generateMipmaps: false,
      depthBuffer: false,
      stencilBuffer: false,
    };

    this.rtA = new THREE.WebGLRenderTarget(VIS_TEX_W, VIS_TEX_H, rtOpts);
    this.rtB = new THREE.WebGLRenderTarget(VIS_TEX_W, VIS_TEX_H, rtOpts);
    for (const rt of [this.rtA, this.rtB]) {
      for (const tex of rt.textures) {
        tex.minFilter = THREE.NearestFilter;
        tex.magFilter = THREE.NearestFilter;
        tex.generateMipmaps = false;
      }
    }

    this.initPosTex = makeFloatDataTexture(data.initPos, VIS_TEX_W, VIS_TEX_H);
    this.initVelTex = makeFloatDataTexture(data.initVel, VIS_TEX_W, VIS_TEX_H);
    this.staticAttrTexture = makeFloatDataTexture(data.staticAttr, VIS_TEX_W, VIS_TEX_H);
    this.staticColorTexture = makeFloatDataTexture(data.staticColor, VIS_TEX_W, VIS_TEX_H);
    this.localInitPosTex = makeFloatDataTexture(data.localInitPos, VIS_TEX_W, VIS_TEX_H);

    this.treePosMassTex = makeFloatDataTexture(
      this.treePosMassData,
      TREE_TEX_WIDTH,
      TREE_TEX_HEIGHT,
    );
    this.treeMetaTex = makeFloatDataTexture(
      this.treeMetaData,
      TREE_TEX_WIDTH,
      TREE_TEX_HEIGHT,
    );

    this.initMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: GPGPU_QUAD_VERT,
      fragmentShader: GPGPU_INIT_FRAG,
      uniforms: {
        uInitPosTex: { value: this.initPosTex },
        uInitVelTex: { value: this.initVelTex },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.advectMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: GPGPU_QUAD_VERT,
      fragmentShader: GPGPU_ADVECT_FRAG,
      uniforms: {
        uPosTex: { value: this.rtA.textures[0] },
        uVelTex: { value: this.rtA.textures[1] },
        uStaticAttrTex: { value: this.staticAttrTexture },
        uTreePosMass: { value: this.treePosMassTex },
        uTreeMeta: { value: this.treeMetaTex },
        uNodeCount: { value: 0 },
        uIsFirstStep: { value: 1 },
        uG: { value: G },
        uThetaSq: { value: THETA_SQ },
        uKickDt: { value: 0 },
        uDriftDt: { value: 0 },
        uStepDt: { value: 0 },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.approachMat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: GPGPU_QUAD_VERT,
      fragmentShader: GPGPU_APPROACH_FRAG,
      uniforms: {
        uLocalInitPosTex: { value: this.localInitPosTex },
        uStaticColorTex: { value: this.staticColorTexture },
        uTimeGyr: { value: 0 },
        uTHandoff: { value: T_HANDOFF },
        uBasisMW: { value: toThreeMat3(BASIS_MW_VIEW) },
        uBasisM31: { value: toThreeMat3(BASIS_M31_VIEW) },
        uMwPos: { value: new THREE.Vector3() },
        uMwVel: { value: new THREE.Vector3() },
        uM31Pos: { value: new THREE.Vector3() },
        uM31Vel: { value: new THREE.Vector3() },
      },
      depthTest: false,
      depthWrite: false,
    });

    this.quadScene = new THREE.Scene();
    this.quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const quadGeom = new THREE.PlaneGeometry(2, 2);
    this.quadMesh = new THREE.Mesh(quadGeom, this.initMat);
    this.quadMesh.frustumCulled = false;
    this.quadScene.add(this.quadMesh);

    // Allocate 24-slice RGBA32F TEXTURE_2D_ARRAY buffers for instant checkpoint scrub
    this.posArrayTex = this.createCheckpointArrayTexture();
    this.velArrayTex = this.createCheckpointArrayTexture();

    // Compile tiny native WebGL2 restore shader program (avoids private Three.js internals)
    const vs = compileShader(gl, gl.VERTEX_SHADER, GPGPU_QUAD_VERT);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, GPGPU_RESTORE_FRAG);
    const prog = gl.createProgram();
    if (!prog) throw new Error('Failed to create restore program');
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, 'position');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`Restore program link error: ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    this.restoreProgram = prog;

    this.restoreLocPosArray = gl.getUniformLocation(prog, 'uPosArray');
    this.restoreLocVelArray = gl.getUniformLocation(prog, 'uVelArray');
    this.restoreLocSlice = gl.getUniformLocation(prog, 'uSlice');

    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    if (!vao || !vbo) throw new Error('Failed to allocate restore VAO/VBO');
    this.restoreVao = vao;
    this.restoreVbo = vbo;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -1, -1, 0,
        1, -1, 0,
        -1, 1, 0,
        1, 1, 0,
      ]),
      gl.STATIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    this.resetToHandoff();
    this.renderer.resetState();
  }

  private createCheckpointArrayTexture(): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture();
    if (!tex) throw new Error('Failed to create TEXTURE_2D_ARRAY');
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    gl.texStorage3D(
      gl.TEXTURE_2D_ARRAY,
      1,
      gl.RGBA32F,
      VIS_TEX_W,
      VIS_TEX_H,
      CHECKPOINT_SLICE_COUNT,
    );
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
    return tex;
  }

  private get activeRT(): THREE.WebGLRenderTarget {
    return this.activeIdx === 0 ? this.rtA : this.rtB;
  }

  private get inactiveRT(): THREE.WebGLRenderTarget {
    return this.activeIdx === 0 ? this.rtB : this.rtA;
  }

  public get posTexture(): THREE.Texture {
    return this.activeRT.textures[0];
  }

  public get velTexture(): THREE.Texture {
    return this.activeRT.textures[1];
  }

  public resetToHandoff(): void {
    const prevRT = this.renderer.getRenderTarget();
    this.activeIdx = 0;
    this.quadMesh.material = this.initMat;
    this.renderer.setRenderTarget(this.rtA);
    this.renderer.render(this.quadScene, this.quadCamera);
    this.renderer.setRenderTarget(prevRT);
  }

  public uploadOctree(
    treePosMass: Float32Array,
    treeMeta: Float32Array,
    nodeCount: number,
  ): void {
    this.currentNodeCount = nodeCount;
    this.treePosMassData.set(treePosMass.subarray(0, nodeCount * 4), 0);
    this.treeMetaData.set(treeMeta.subarray(0, nodeCount * 4), 0);
    this.treePosMassTex.needsUpdate = true;
    this.treeMetaTex.needsUpdate = true;
  }

  public step(
    kickDt: number,
    driftDt: number,
    stepDt: number,
    isFirstStep: boolean,
  ): void {
    const srcRT = this.activeRT;
    const dstRT = this.inactiveRT;
    const prevRT = this.renderer.getRenderTarget();

    this.advectMat.uniforms.uPosTex.value = srcRT.textures[0];
    this.advectMat.uniforms.uVelTex.value = srcRT.textures[1];
    this.advectMat.uniforms.uNodeCount.value = this.currentNodeCount;
    this.advectMat.uniforms.uIsFirstStep.value = isFirstStep ? 1 : 0;
    this.advectMat.uniforms.uKickDt.value = kickDt;
    this.advectMat.uniforms.uDriftDt.value = driftDt;
    this.advectMat.uniforms.uStepDt.value = stepDt;

    this.quadMesh.material = this.advectMat;
    this.renderer.setRenderTarget(dstRT);
    this.renderer.render(this.quadScene, this.quadCamera);
    this.renderer.setRenderTarget(prevRT);

    this.activeIdx = 1 - this.activeIdx;
  }

  public setApproachState(
    tGyr: number,
    mwPos: Vec3,
    mwVel: Vec3,
    m31Pos: Vec3,
    m31Vel: Vec3,
  ): void {
    const prevRT = this.renderer.getRenderTarget();
    const u = this.approachMat.uniforms;
    u.uTimeGyr.value = tGyr;
    (u.uMwPos.value as THREE.Vector3).set(mwPos[0], mwPos[1], mwPos[2]);
    (u.uMwVel.value as THREE.Vector3).set(mwVel[0], mwVel[1], mwVel[2]);
    (u.uM31Pos.value as THREE.Vector3).set(m31Pos[0], m31Pos[1], m31Pos[2]);
    (u.uM31Vel.value as THREE.Vector3).set(m31Vel[0], m31Vel[1], m31Vel[2]);

    this.quadMesh.material = this.approachMat;
    this.renderer.setRenderTarget(this.activeRT);
    this.renderer.render(this.quadScene, this.quadCamera);
    this.renderer.setRenderTarget(prevRT);
  }

  public saveCheckpoint(slice: number): void {
    if (slice < 0 || slice >= CHECKPOINT_SLICE_COUNT) return;
    const gl = this.gl;
    const prevRT = this.renderer.getRenderTarget();

    this.renderer.setRenderTarget(this.activeRT);
    const fbo = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING) as WebGLFramebuffer | null;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fbo);

    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.posArrayTex);
    gl.copyTexSubImage3D(
      gl.TEXTURE_2D_ARRAY,
      0,
      0,
      0,
      slice,
      0,
      0,
      VIS_TEX_W,
      VIS_TEX_H,
    );

    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.velArrayTex);
    gl.copyTexSubImage3D(
      gl.TEXTURE_2D_ARRAY,
      0,
      0,
      0,
      slice,
      0,
      0,
      VIS_TEX_W,
      VIS_TEX_H,
    );

    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);

    this.renderer.setRenderTarget(prevRT);
    this.renderer.resetState();
  }

  public restoreCheckpoint(slice: number): void {
    if (slice < 0 || slice >= CHECKPOINT_SLICE_COUNT) return;
    const gl = this.gl;
    const prevRT = this.renderer.getRenderTarget();

    this.renderer.setRenderTarget(this.activeRT);

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);
    gl.viewport(0, 0, VIS_TEX_W, VIS_TEX_H);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);

    gl.useProgram(this.restoreProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.posArrayTex);
    gl.uniform1i(this.restoreLocPosArray, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.velArrayTex);
    gl.uniform1i(this.restoreLocVelArray, 1);

    gl.uniform1i(this.restoreLocSlice, slice);

    gl.bindVertexArray(this.restoreVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);

    this.renderer.setRenderTarget(prevRT);
    this.renderer.resetState();
  }

  public dispose(): void {
    const gl = this.gl;
    this.rtA.dispose();
    this.rtB.dispose();
    this.initPosTex.dispose();
    this.initVelTex.dispose();
    this.staticAttrTexture.dispose();
    this.staticColorTexture.dispose();
    this.localInitPosTex.dispose();
    this.treePosMassTex.dispose();
    this.treeMetaTex.dispose();
    this.initMat.dispose();
    this.advectMat.dispose();
    this.approachMat.dispose();
    this.quadMesh.geometry.dispose();
    gl.deleteTexture(this.posArrayTex);
    gl.deleteTexture(this.velArrayTex);
    gl.deleteProgram(this.restoreProgram);
    gl.deleteVertexArray(this.restoreVao);
    gl.deleteBuffer(this.restoreVbo);
  }
}
