# Soccer360 - Claude Code Context

## What This Project Is

Automated 360 soccer video processing pipeline. Takes equirectangular 360 match footage and produces broadcast-style auto-follow video, tactical wide-angle view, and highlight clips. Runs on bare-metal Ubuntu 22.04 with Tesla P40 GPU inside Docker.

## Tech Stack

- **Python 3.11**, packaged via `pyproject.toml`, installed as `soccer360` CLI
- **YOLO (ultralytics)** for ball detection on GPU
- **py360convert** for equirectangular-to-perspective projection
- **filterpy** for Kalman filtering (tracker + camera smoothing)
- **ffmpeg** for all video I/O (streaming pipes, never raw frames to disk)
- **Docker** (NVIDIA CUDA 12.2 base image) for deployment
- No web framework, no database -- pure batch video processing pipeline

## Architecture

Two-pass streaming pipeline with 8 phases:

1. **Detector** (`src/detector.py`): GPU YOLO inference + Field-of-Interest filtering
2. **Tracker** (`src/tracker.py`): ByteTrack two-stage association + ball selection
3. **Camera** (`src/camera.py`): Kalman + EMA smoothing -> per-frame yaw/pitch/FOV
4. **Reframer** (`src/reframer.py`): py360convert e2p, 12 parallel segment workers
5. **Reframer** (tactical): Fixed wide-angle view, same parallel strategy
6. **Highlights** (`src/highlights.py`): Heuristic event detection + clip export
7. **Exporter** (`src/exporter.py`): Output organization + metadata
8. **Pipeline** (`src/pipeline.py`): Orchestrator coordinating all phases

Key data flow: ffmpeg pipe -> numpy arrays -> GPU inference -> JSONL -> JSON -> ffmpeg pipe -> MP4. No intermediate frame files.

## Code Conventions

- All source in `src/`, tests in `tests/`
- Config: `configs/pipeline.yaml` (runtime), `configs/model_config.yaml` (training)
- Ruff for linting, target Python 3.11, line length 100
- Type hints with `from __future__ import annotations`
- Logging via `logging.getLogger("soccer360.<module>")`
- Detection format: JSONL with `{frame, bbox: [x1,y1,x2,y2], confidence, class}`
- All pixel-space thresholds are in `detector.resolution` coordinate space (default 1920x960)

## Important Math

- **Equirectangular mapping**: `yaw = (x/W)*360 - 180`, `pitch = 90 - (y/H)*180`
- **Angle wrapping**: `(-180, 180]` half-open interval via `a % 360; if a > 180: a -= 360`
- **Vertical FOV**: Tangent-based `fov_v = degrees(2 * atan(tan(radians(fov_h/2)) * (h/w)))`, NOT linear ratio
- **FoI auto-center**: Yaw histogram (5-degree bins) -> peak -> circular mean of peak +/- 1 bin

## Running Tests

No local Python environment -- project runs in Docker:

```bash
docker compose run --rm worker pytest tests/ -v
```

For quick syntax checking locally:

```bash
python3 -c "import ast; ast.parse(open('src/detector.py').read())"
```

## Key Files to Read First

1. `configs/pipeline.yaml` -- All tunable parameters
2. `src/pipeline.py` -- Orchestrator, shows the 8-phase flow
3. `src/detector.py` -- Detection + FoI filtering (most complex module)
4. `src/camera.py` -- Camera smoothing pipeline (Kalman + EMA + deadband)
5. `src/utils.py` -- Shared helpers (FFmpeg I/O, angle math, config loading)

## Common Tasks

**Add a new pipeline phase:** Edit `src/pipeline.py` run method, add the phase between existing ones. Each phase reads from the previous phase's output file in the work directory.

**Change detection behavior:** Edit `src/detector.py`. FoI config is in `field_of_interest` section. Detection resolution and confidence thresholds in `detector` section.

**Tune camera smoothing:** Edit `camera` section in `configs/pipeline.yaml`. Key params: `ema_alpha` (lower = smoother), `deadband_deg`, `max_pan_speed_deg_per_sec`.

**Add a new config parameter:** Add to `configs/pipeline.yaml`, read it in the relevant module's `__init__`, add to `tests/conftest.py` test config.

## Do NOT

- Run `pytest` directly (no local venv, use Docker)
- Modify frame I/O to write intermediate frames to disk (streaming design is intentional)
- Change the `(-180, 180]` angle convention -- camera.py and detector.py both depend on it
- Use linear ratio for vertical FOV computation (must use tangent formula)
- Skip adding new config params to `tests/conftest.py` test config fixture
