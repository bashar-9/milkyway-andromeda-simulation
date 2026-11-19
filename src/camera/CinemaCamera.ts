import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { M31_PARAMS, MW_PARAMS, Vec3 } from '../physics/constants.ts';

export type CameraFocus = 'COM' | 'MW' | 'M31';

const MW_MASS_FRAC = MW_PARAMS.mTotal / (MW_PARAMS.mTotal + M31_PARAMS.mTotal);
const M31_MASS_FRAC = 1.0 - MW_MASS_FRAC;

export class CinemaCamera {
  public readonly camera: THREE.PerspectiveCamera;
  public readonly controls: OrbitControls;

  private focus: CameraFocus = 'COM';
  private cinemaDrift = true;
  private transitionAlpha = 0;

  private readonly lastFocusTarget = new THREE.Vector3();
  private readonly desiredFocusTarget = new THREE.Vector3();
  private readonly desiredResetCamPos = new THREE.Vector3(18, 118, 295);
  private hasInitializedTarget = false;
  private resetting = false;

  private readonly onCinemaChanged?: (enabled: boolean) => void;
  private readonly onFocusChanged?: (focus: CameraFocus) => void;

  constructor(
    domElement: HTMLElement,
    aspect: number,
    callbacks?: {
      onCinemaChanged?: (enabled: boolean) => void;
      onFocusChanged?: (focus: CameraFocus) => void;
    },
  ) {
    this.onCinemaChanged = callbacks?.onCinemaChanged;
    this.onFocusChanged = callbacks?.onFocusChanged;

    this.camera = new THREE.PerspectiveCamera(42, aspect, 0.2, 8000);
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

    // Yield immediately to user interaction
    this.controls.addEventListener('start', () => {
      this.resetting = false;
      if (this.cinemaDrift) {
        this.cinemaDrift = false;
        this.onCinemaChanged?.(false);
      }
    });
  }

  public getFocus(): CameraFocus {
    return this.focus;
  }

  public isCinemaDrift(): boolean {
    return this.cinemaDrift;
  }

  public setFocus(focus: CameraFocus): void {
    this.focus = focus;
    this.transitionAlpha = 1.0;
    this.onFocusChanged?.(focus);
  }

  public cycleFocus(): CameraFocus {
    const order: CameraFocus[] = ['COM', 'MW', 'M31'];
    const next = order[(order.indexOf(this.focus) + 1) % order.length];
    this.setFocus(next);
    return next;
  }

  public setCinemaDrift(enabled: boolean): void {
    this.cinemaDrift = enabled;
    this.onCinemaChanged?.(enabled);
  }

  public toggleCinemaDrift(): boolean {
    this.setCinemaDrift(!this.cinemaDrift);
    return this.cinemaDrift;
  }

  public reset(): void {
    this.focus = 'COM';
    this.transitionAlpha = 1.0;
    this.resetting = true;
    this.onFocusChanged?.('COM');
  }

  public setAspect(aspect: number): void {
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  public update(
    dtSec: number,
    mwCenter: Vec3,
    m31Center: Vec3,
  ): void {
    const dt = Math.min(0.05, Math.max(0.0001, dtSec));

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
        // Keep camera offset relative to lerping target when focusing a specific galaxy
        const shift = this.controls.target.clone().sub(prevT);
        this.camera.position.add(shift);
      }
      this.transitionAlpha *= Math.exp(-dt * 2.5);
    }

    // Subtle sweeping cinema orbit around target
    if (this.cinemaDrift && !this.resetting) {
      const offset = this.camera.position.clone().sub(this.controls.target);
      const angle = dt * 0.011;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const nx = offset.x * cosA - offset.z * sinA;
      const nz = offset.x * sinA + offset.z * cosA;
      offset.x = nx;
      offset.z = nz;
      this.camera.position.copy(this.controls.target).add(offset);
    }

    this.controls.update();
  }

  public dispose(): void {
    this.controls.dispose();
  }
}
