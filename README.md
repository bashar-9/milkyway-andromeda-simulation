# Milky Way × Andromeda — 3D N-Body Simulation

A real-time, physically grounded 3D WebGL2 simulation of the Milky Way and Andromeda (M31) galactic collision, running a Barnes-Hut macro N-body solver in a Web Worker alongside GPGPU ping-pong advection for **196,608** visual star, nebula, and dust tracers.

- **Live Demo:** https://bashar-9.github.io/milkyway-andromeda-simulation/
- **Pre-rendered 1080p60 Video:** Download `milky-way-andromeda-collision.mp4` from the [v1.0.0 Release](https://github.com/bashar-9/milkyway-andromeda-simulation/releases/tag/v1.0.0), or render locally with `npm run export-video`.

## Features

- **Self-Consistent 3D N-Body Physics:** 6,144 macro particles (dark matter halo, exponential disk, and Hernquist bulge for both galaxies) integrated with a leapfrog Kick-Drift-Kick (KDK) scheme and a Barnes-Hut octree (`θ = 0.75`) inside a dedicated Web Worker (`src/physics/nbodyWorker.ts`).
- **Observational Initial Conditions:** Calibrated to HST/Gaia proper-motion parameters (van der Marel et al. 2012) starting from `D₀ = 770 kpc`, `v_rad = -109.3 km/s`, `v_tan = 28.0 km/s`, with an analytic RK4 approach phase handing off cleanly to full N-body at `T = 2.60 Gyr` and merging into Milkomeda through `T = 7.50 Gyr`.
- **GPGPU Visual Advection:** 196,608 visual particles (`512 × 384` floating-point ping-pong textures) advected on the GPU directly against the Worker-uploaded octree:
  - `140,000` stars (OB, disk, bulge, and halo/globular cluster populations)
  - `32,000` gas continuum & H-II nebula tracers
  - `24,608` dust filaments
- **Director's Camera & Manual Orbit:** Automated multi-shot choreography (`src/camera/DirectorChoreography.ts`) that smoothly yields the moment you interact with `OrbitControls`.
- **Timeline Scrubbing & Checkpoints:** 24 VRAM + Worker state checkpoints across the merger so you can scrub backward and forward across cosmic epochs (`1st Impact`, `Tidal Tails`, `2nd Impact`, `Coalescence`, `Milkomeda`).
- **Offline 1080p60 Video Exporter:** Deterministic frame-by-frame capture via headless Chrome/Edge piped directly to `ffmpeg` (`scripts/export-video.mjs`).

## Prerequisites

- **Node.js** `>= 18` and **npm**
- A modern browser with **WebGL2** support
- *(Optional, for `npm run export-video`)* **FFmpeg** on `PATH` and **Google Chrome** / **Chromium** / **Microsoft Edge**

## Quickstart

```bash
git clone https://github.com/bashar-9/milkyway-andromeda-simulation.git
cd milkyway-andromeda-simulation
npm install
npm run dev
```

Open the local URL printed by Vite (defaults to `http://localhost:5173`).

### Other Scripts

```bash
# Type-check and build production bundle to dist/
npm run build

# Preview the production build locally
npm run preview

# Run the standalone N-body conservation & physics verification suite
npm run verify-physics
```

## Controls

| Input | Action |
| --- | --- |
| `Space` | Play / Pause (restarts from start if at `T_MAX`) |
| `1` / `2` / `3` | Focus camera on `COM` / `MW` / `M31` |
| `C` or `D` | Toggle Director's Camera |
| `R` | Reset camera |
| `H` | Hide / Show HUD |
| Mouse drag / scroll / right-drag | Orbit / Zoom / Pan (automatically switches out of Director mode) |
| Timeline bar / epoch ticks | Scrub or jump directly to merger epochs |

## Offline 1080p60 Video Export

Build `dist/` first, then run the exporter:

```bash
npm run build
npm run export-video
```

Common flags:

```bash
node scripts/export-video.mjs \
  --out milky-way-andromeda-collision.mp4 \
  --width 1920 \
  --height 1080 \
  --fps 60 \
  --gyr-per-sec 0.12 \
  --start-gyr 2.84 \
  --crf 15 \
  --preset slow
```

- Pass `--test-frames 120` for a quick 2-second test render.
- Set `CHROME_PATH=/path/to/chrome` if the exporter cannot auto-detect your browser binary.

## Project Structure

```text
src/
├── camera/        # OrbitControls wrapper and Director's Camera choreography
├── export/        # Deterministic in-browser frame stepping for headless export
├── gpu/           # WebGL2 ping-pong GPGPU advection and tracer generation
├── physics/       # Constants, galaxy profiles, Barnes-Hut octree, Web Worker, verifier
├── render/        # Three.js post-processing pipeline and custom GLSL shaders
├── ui/            # Minimal glass HUD and telemetry readout
├── main.ts        # Application bootstrap, worker sync, and animation loop
└── style.css      # HUD and viewport styling
scripts/
└── export-video.mjs  # Headless browser + FFmpeg 1080p60 export orchestrator
```

## Contributing

Issues and pull requests are welcome! Please make sure `npm run build` and `npm run verify-physics` pass before opening a PR.

## License

[MIT](LICENSE)
