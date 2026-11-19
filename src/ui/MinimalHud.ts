import { CameraFocus } from '../camera/CinemaCamera.ts';
import { EPOCH_MARKERS, T_INITIAL_VIEW, T_MAX } from '../physics/constants.ts';

export interface HudCallbacks {
  onTogglePlay: () => void;
  onCycleSpeed: () => void;
  onSeekTime: (timeGyr: number) => void;
  onCycleFocus: () => void;
  onToggleDirector: () => void;
  onResetCamera: () => void;
}

const PLAY_SVG =
  '<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.8v10.4a.6.6 0 0 0 .92.51l8.4-5.2a.6.6 0 0 0 0-1.02l-8.4-5.2A.6.6 0 0 0 4 2.8z"/></svg>';
const PAUSE_SVG =
  '<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><rect x="3.5" y="2.5" width="3.2" height="11" rx="1"/><rect x="9.3" y="2.5" width="3.2" height="11" rx="1"/></svg>';
const RESET_SVG =
  '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 8a5.5 5.5 0 1 0 1.4-3.65L2.5 5.8"/><path d="M2.5 2.5v3.3h3.3"/></svg>';
const EYE_OFF_SVG =
  '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><path d="M1.5 8s2.4-4.2 6.5-4.2S14.5 8 14.5 8s-2.4 4.2-6.5 4.2S1.5 8 1.5 8z"/><circle cx="8" cy="8" r="1.9"/></svg>';

export function getEpochLabel(timeGyr: number, sepKpc: number): string {
  if (timeGyr < 2.6) return 'APPROACH';
  if (timeGyr < 3.65) return 'HALO INTERACTION';
  if (timeGyr < 4.02) return 'FIRST IMPACT';
  if (timeGyr < 5.05) return 'TIDAL TAILS';
  if (timeGyr < 5.55) return 'SECOND PASSAGE';
  if (sepKpc > 10.5 && timeGyr < 6.1) return 'COALESCENCE';
  return 'MILKOMEDA';
}

export class MinimalHud {
  private readonly root: HTMLElement;
  private readonly timeEl: HTMLElement;
  private readonly sepEl: HTMLElement;
  private readonly vrelEl: HTMLElement;
  private readonly epochEl: HTMLElement;
  private readonly bootSplashEl: HTMLElement;

  private readonly playBtn: HTMLButtonElement;
  private readonly speedBtn: HTMLButtonElement;
  private readonly trackWrap: HTMLElement;
  private readonly cachedBar: HTMLElement;
  private readonly fillBar: HTMLElement;
  private readonly thumbEl: HTMLElement;
  private readonly tickEls: Array<{ el: HTMLElement; timeGyr: number }> = [];

  private readonly focusBtn: HTMLButtonElement;
  private readonly directorBtn: HTMLButtonElement;
  private readonly resetBtn: HTMLButtonElement;
  private readonly hideBtn: HTMLButtonElement;
  private readonly restoreChip: HTMLButtonElement;

  private dragging = false;
  private hudHidden = false;

  constructor(root: HTMLElement, callbacks: HudCallbacks) {
    this.root = root;

    // Top-left whisper-thin telemetry
    const telemetry = document.createElement('div');
    telemetry.className = 'telemetry';
    telemetry.innerHTML = `
      <div class="telemetry-title">MILKY WAY &times; ANDROMEDA</div>
      <div class="telemetry-readout">
        <span data-field="time">T +0.00 GYR</span>
        <span class="telemetry-sep">&middot;</span>
        <span data-field="sep">SEP 770 KPC</span>
        <span class="telemetry-sep">&middot;</span>
        <span data-field="vrel">V<sub>REL</sub> 113 KM/S</span>
        <span class="telemetry-sep">&middot;</span>
        <span class="telemetry-epoch" data-field="epoch">APPROACH</span>
      </div>
    `;
    this.timeEl = telemetry.querySelector('[data-field="time"]')!;
    this.sepEl = telemetry.querySelector('[data-field="sep"]')!;
    this.vrelEl = telemetry.querySelector('[data-field="vrel"]')!;
    this.epochEl = telemetry.querySelector('[data-field="epoch"]')!;

    // Brief boot splash
    this.bootSplashEl = document.createElement('div');
    this.bootSplashEl.className = 'boot-splash';
    this.bootSplashEl.textContent = 'Initializing self-consistent N-body gravity...';

    // Bottom-center floating glass timeline pill
    const pill = document.createElement('div');
    pill.className = 'timeline-pill';

    this.playBtn = document.createElement('button');
    this.playBtn.className = 'hud-btn icon-only';
    this.playBtn.title = 'Play / Pause (Space)';
    this.playBtn.innerHTML = PAUSE_SVG;
    this.playBtn.addEventListener('click', callbacks.onTogglePlay);

    this.speedBtn = document.createElement('button');
    this.speedBtn.className = 'hud-btn mono';
    this.speedBtn.title = 'Simulation Speed';
    this.speedBtn.textContent = '1×';
    this.speedBtn.addEventListener('click', callbacks.onCycleSpeed);

    this.trackWrap = document.createElement('div');
    this.trackWrap.className = 'track-wrap';
    const rail = document.createElement('div');
    rail.className = 'track-rail';

    this.cachedBar = document.createElement('div');
    this.cachedBar.className = 'track-cached';
    this.fillBar = document.createElement('div');
    this.fillBar.className = 'track-fill';
    this.thumbEl = document.createElement('div');
    this.thumbEl.className = 'track-thumb';

    rail.append(this.cachedBar, this.fillBar, this.thumbEl);

    const timelineSpan = Math.max(1e-4, T_MAX - T_INITIAL_VIEW);

    for (const marker of EPOCH_MARKERS) {
      const tick = document.createElement('div');
      tick.className = 'epoch-tick';
      const pct = ((marker.timeGyr - T_INITIAL_VIEW) / timelineSpan) * 100;
      tick.style.left = `${pct}%`;

      const dot = document.createElement('div');
      dot.className = 'epoch-dot';
      const tip = document.createElement('div');
      tip.className = 'epoch-tooltip';
      tip.textContent = `${marker.label} · +${marker.timeGyr.toFixed(2)} Gyr`;
      tick.append(dot, tip);

      tick.addEventListener('pointerdown', (e) => {
        e.stopPropagation();
        callbacks.onSeekTime(marker.timeGyr);
      });

      rail.appendChild(tick);
      this.tickEls.push({ el: tick, timeGyr: marker.timeGyr });
    }
    this.trackWrap.appendChild(rail);

    const seekFromPointer = (clientX: number): void => {
      const rect = this.trackWrap.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width)));
      callbacks.onSeekTime(T_INITIAL_VIEW + ratio * timelineSpan);
    };

    this.trackWrap.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      this.trackWrap.setPointerCapture(e.pointerId);
      seekFromPointer(e.clientX);
    });
    this.trackWrap.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      seekFromPointer(e.clientX);
    });
    const endDrag = (): void => {
      this.dragging = false;
    };
    this.trackWrap.addEventListener('pointerup', endDrag);
    this.trackWrap.addEventListener('pointercancel', endDrag);

    const div1 = document.createElement('div');
    div1.className = 'hud-divider';

    this.focusBtn = document.createElement('button');
    this.focusBtn.className = 'hud-btn mono';
    this.focusBtn.title = 'Focus Target (1 / 2 / 3)';
    this.focusBtn.textContent = 'COM';
    this.focusBtn.addEventListener('click', callbacks.onCycleFocus);

    this.directorBtn = document.createElement('button');
    this.directorBtn.className = 'hud-btn active';
    this.directorBtn.title = "Toggle Director's Camera (D)";
    this.directorBtn.textContent = 'Director';
    this.directorBtn.addEventListener('click', callbacks.onToggleDirector);

    this.resetBtn = document.createElement('button');
    this.resetBtn.className = 'hud-btn icon-only';
    this.resetBtn.title = 'Reset Camera (R)';
    this.resetBtn.innerHTML = RESET_SVG;
    this.resetBtn.addEventListener('click', callbacks.onResetCamera);

    this.hideBtn = document.createElement('button');
    this.hideBtn.className = 'hud-btn icon-only';
    this.hideBtn.title = 'Hide / Show HUD (H)';
    this.hideBtn.innerHTML = EYE_OFF_SVG;
    this.hideBtn.addEventListener('click', () => this.toggleHudVisibility());

    pill.append(
      this.playBtn,
      this.speedBtn,
      this.trackWrap,
      div1,
      this.focusBtn,
      this.directorBtn,
      this.resetBtn,
      this.hideBtn,
    );

    this.restoreChip = document.createElement('button');
    this.restoreChip.className = 'hud-restore-chip';
    this.restoreChip.textContent = 'Show HUD (H)';
    this.restoreChip.addEventListener('click', () => this.toggleHudVisibility());

    this.root.append(telemetry, this.bootSplashEl, pill, this.restoreChip);
  }

  public setBootReady(): void {
    this.bootSplashEl.classList.add('hidden');
  }

  public setPlaying(playing: boolean): void {
    this.playBtn.innerHTML = playing ? PAUSE_SVG : PLAY_SVG;
  }

  public setSpeedLabel(label: string): void {
    this.speedBtn.textContent = label;
  }

  public setFocus(focus: CameraFocus): void {
    this.focusBtn.textContent = focus;
  }

  public setDirector(enabled: boolean): void {
    this.directorBtn.classList.toggle('active', enabled);
  }

  public toggleHudVisibility(): boolean {
    this.hudHidden = !this.hudHidden;
    this.root.classList.toggle('hud-hidden', this.hudHidden);
    this.restoreChip.classList.toggle('visible', this.hudHidden);
    return this.hudHidden;
  }

  public update(
    timeGyr: number,
    maxCachedTimeGyr: number,
    sepKpc: number,
    relSpeedKms: number,
    isSeeking: boolean,
  ): void {
    const timelineSpan = Math.max(1e-4, T_MAX - T_INITIAL_VIEW);
    const pct = Math.max(0, Math.min(100, ((timeGyr - T_INITIAL_VIEW) / timelineSpan) * 100));
    const cachedPct = Math.max(
      0,
      Math.min(100, ((maxCachedTimeGyr - T_INITIAL_VIEW) / timelineSpan) * 100),
    );

    this.fillBar.style.width = `${pct.toFixed(3)}%`;
    this.cachedBar.style.width = `${cachedPct.toFixed(3)}%`;
    this.thumbEl.style.left = `${pct.toFixed(3)}%`;
    this.thumbEl.classList.toggle('seeking', isSeeking);

    for (const tick of this.tickEls) {
      tick.el.classList.toggle('passed', timeGyr >= tick.timeGyr - 0.02);
    }

    this.timeEl.textContent = `T +${timeGyr.toFixed(2)} GYR`;
    this.sepEl.textContent = `SEP ${Math.round(sepKpc)} KPC`;
    this.vrelEl.innerHTML = `V<sub>REL</sub> ${Math.round(relSpeedKms)} KM/S`;
    this.epochEl.textContent = getEpochLabel(timeGyr, sepKpc);
  }
}
