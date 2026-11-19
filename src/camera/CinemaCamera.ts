import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { M31_PARAMS, MW_PARAMS, Vec3 } from '../physics/constants.ts';
import { DirectorPose, evaluateDirectorShot } from './DirectorChoreography.ts';

export type CameraFocus = 'COM' | 'MW' | 'M31';

const MW_MASS_FRAC = MW_PARAMS.mTotal / (MW_PARAMS.mTotal + M31_PARAMS.mTotal);
const M31_MASS_FRAC = 1.0 - MW_MASS_FRAC;
const DEFAULT_MANUAL_FOV = 42.0;

export class CinemaCamera {
  public readonly camera: THREE.PerspectiveCamera;
  public readonly controls: OrbitControls;

  private focus: CameraFocus = 'COM';
  private directorMode = true;
  private hasSnappedDirector = false;
  private transitionAlpha = 0;

  private currentFov = DEFAULT_MANUAL_FOV;
  private currentRollRad = 0;

  private readonly lastFocusTarget = new THREE.Vector3();
  private readonly desiredFocusTarget = new THREE.Vector3();
  private readonly desiredResetCamPos = new THREE.Vector3(18, 118, 295);
  private readonly directorPose: DirectorPose = {
    position: new THREE.Vector3(),
    target: new THREE.Vector3(),
    fov: DEFAULT_MANUAL_FOV,
    rollRad: 0,
  };
  private hasInitializedTarget = false;
  private resetting = false;

  private readonly onDirectorChanged?: (enabled: boolean) => void;
  private readonly onFocusChanged?: (focus: CameraFocus) => void;

  constructor(
    domElement: HTMLElement,
    aspect: number,
    callbacks?: {
      onDirectorChanged?: (enabled: boolean) => void;
      onFocusChanged?: (focus: CameraFocus) => void;
    },
  ) {
    this.onDirectorChanged = callbacks?.onDirectorChanged;
    this.onFocusChanged = callbacks?.onFocusChanged;

    this.camera = new THREE.PerspectiveCamera(DEFAULT_MANUAL_FOV, aspect, 0.2, 8000);
    this.camera.position.copy(this.desiredResetCamPos);

    this.controls = new OrbitControls(this.camera, domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.055;
    this.controls.minDistance = 4.0;
    this.controls.maxDistance = 1200.0;
    this.controls.rotateSpeed = 0.72;
    this.controls.zoomSpeed = 0.95;
    this.controls.panSpeed = 0.75;
    this.controls.target.set(0, 0, 0);

    // Yield immediately to user interaction without jarring jump
    this.controls.addEventListener('start', () => {
      this.resetting = false;
      if (this.directorMode) {
        this.directorMode = false;
        this.camera.up.set(0, 1, 0);
        this.onDirectorChanged?.(false);
      }
    });
  }

  public getFocus(): CameraFocus {
    return this.focus;
  }

  public isDirectorMode(): boolean {
    return this.directorMode;
  }

  public setFocus(focus: CameraFocus): void {
    this.focus = focus;
    this.transitionAlpha = 1.0;
    this.resetting = false;
    if (this.directorMode) {
      this.directorMode = false;
      this.camera.up.set(0, 1, 0);
      this.onDirectorChanged?.(false);
    }
    this.onFocusChanged?.(focus);
  }

  public cycleFocus(): CameraFocus {
    const order: CameraFocus[] = ['COM', 'MW', 'M31'];
    const next = order[(order.indexOf(this.focus) + 1) % order.length];
    this.setFocus(next);
    return next;
  }

  public setDirectorMode(enabled: boolean): void {
    this.directorMode = enabled;
    if (enabled) {
      this.resetting = false;
      this.focus = 'COM';
      this.onFocusChanged?.('COM');
    } else {
      this.camera.up.set(0, 1, 0);
    }
    this.onDirectorChanged?.(enabled);
  }

  public toggleDirectorMode(): boolean {
    this.setDirectorMode(!this.directorMode);
    return this.directorMode;
  }

  public snapToDirector(
    timeGyr: number,
    mwCenter: Vec3,
    m31Center: Vec3,
    sepKpc: number,
    realTimeSec: number,
  ): void {
    evaluateDirectorShot(timeGyr, mwCenter, m31Center, sepKpc, realTimeSec, this.directorPose);
    this.camera.position.copy(this.directorPose.position);
    this.controls.target.copy(this.directorPose.target);
    this.lastFocusTarget.copy(this.directorPose.target);
    this.hasInitializedTarget = true;
    this.currentFov = this.directorPose.fov;
    this.currentRollRad = this.directorPose.rollRad;
    this.hasSnappedDirector = true;
    this.applyDirectorOrientation();
  }

  public reset(): void {
    this.focus = 'COM';
    this.directorMode = false;
    this.transitionAlpha = 1.0;
    this.resetting = true;
    this.camera.up.set(0, 1, 0);
    this.onDirectorChanged?.(false);
    this.onFocusChanged?.('COM');
  }

  public setAspect(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  private applyDirectorOrientation(): void {
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(this.controls.target);
    if (Math.abs(this.currentRollRad) > 1e-5) {
      this.camera.rotateZ(this.currentRollRad);
    }
    if (Math.abs(this.currentFov - this.camera.fov) > 0.01) {
      this.camera.fov = this.currentFov;
      this.camera.updateProjectionMatrix();
    }
  }

  public update(
    dtSec: number,
    timeGyr: number,
    mwCenter: Vec3,
    m31Center: Vec3,
    sepKpc: number,
    realTimeSec: number,
    isSeeking: boolean = false,
  ): void {
    const dt = Math.min(0.05, Math.max(0.0001, dtSec));

    if (this.directorMode) {
      evaluateDirectorShot(timeGyr, mwCenter, m31Center, sepKpc, realTimeSec, this.directorPose);

      if (!this.hasSnappedDirector) {
        this.camera.position.copy(this.directorPose.position);
        this.controls.target.copy(this.directorPose.target);
        this.currentFov = this.directorPose.fov;
        this.currentRollRad = this.directorPose.rollRad;
        this.hasSnappedDirector = true;
      } else {
        const rate = isSeeking ? 4.2 : 2.7;
        const k = 1.0 - Math.exp(-dt * rate);
        this.camera.position.lerp(this.directorPose.position, k);
        this.controls.target.lerp(this.directorPose.target, k);
        this.currentFov += (this.directorPose.fov - this.currentFov) * k;
        this.currentRollRad += (this.directorPose.rollRad - this.currentRollRad) * k;
      }

      this.lastFocusTarget.copy(this.controls.target);
      this.hasInitializedTarget = true;
      this.applyDirectorOrientation();
      return;
    }

    // Manual OrbitControls mode
    if (this.focus === 'MW') {
      this.desiredFocusTarget.set(mwCenter[0], mwCenter[1], mwCenter[2]);
    } else if (this.focus === 'M31') {
      this.desiredFocusTarget.set(m31Center[0], m31Center[1], m31Center[2]);
    } else {
      this.desiredFocusTarget.set(
        MW_MASS_FRAC * mwCenter[0] + M31_MASS_FRAC * m31Center[0],
        MW_MASS_FRAC * mwCenter[1] + M31_MASS_FRAC * m31Center[1],
        MW_MASS_FRAC * mwCenter[2] + M31_MASS_FRAC * m31Center[2],
      );
    }

    if (!this.hasInitializedTarget) {
      this.lastFocusTarget.copy(this.desiredFocusTarget);
      this.controls.target.copy(this.desiredFocusTarget);
      this.hasInitializedTarget = true;
    }

    // Track moving galaxy core cleanly without clobbering user's pan offset
    const dx = this.desiredFocusTarget.x - this.lastFocusTarget.x;
    const dy = this.desiredFocusTarget.y - this.lastFocusTarget.y;
    const dz = this.desiredFocusTarget.z - this.lastFocusTarget.z;
    this.controls.target.x += dx;
    this.controls.target.y += dy;
    this.controls.target.z += dz;
    this.camera.position.x += dx;
    this.camera.position.y += dy;
    this.camera.position.z += dz;
    this.lastFocusTarget.copy(this.desiredFocusTarget);

    // Smoothly glide target to exact core when switching focus or resetting
    if (this.transitionAlpha > 0.002) {
      const k = 1.0 - Math.exp(-dt * 5.0);
      const prevT = this.controls.target.clone();
      this.controls.target.lerp(this.desiredFocusTarget, k);
      if (this.resetting) {
        this.camera.position.lerp(this.desiredResetCamPos, k);
        if (this.camera.position.distanceToSquared(this.desiredResetCamPos) < 0.25) {
          this.resetting = false;
        }
      } else {
        const shift = this.controls.target.clone().sub(prevT);
        this.camera.position.add(shift);
      }
      this.transitionAlpha *= Math.exp(-dt * 2.5);
    }

    // Smoothly level roll & reset FOV back to 42 deg when returning to manual control
    this.currentRollRad *= Math.exp(-dt * 6.0);
    if (Math.abs(DEFAULT_MANUAL_FOV - this.currentFov) > 0.01) {
      this.currentFov += (DEFAULT_MANUAL_FOV - this.currentFov) * (1.0 - Math.exp(-dt * 4.5));
      this.camera.fov = this.currentFov;
      this.camera.updateProjectionMatrix();
    }

    this.camera.up.set(0, 1, 0);
    this.controls.update();
    if (Math.abs(this.currentRollRad) > 1e-4) {
      this.camera.rotateZ(this.currentRollRad);
    }
  }

  public dispose(): void {
    this.controls.dispose();
  }
}
