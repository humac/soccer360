# Soccer360 - AI Agent Context

This file provides context for AI coding agents (Claude Code, Copilot, Cursor, Windsurf, etc.) working on this codebase.

## Project Overview

Soccer360 is an automated 360 soccer video processing pipeline. It takes equirectangular 360 match footage from an Insta360 camera placed at midfield and produces:
- **broadcast.mp4**: Auto-follow broadcast-style video tracking the ball
- **tactical_wide.mp4**: Fixed wide-angle tactical overview
- **highlight_*.mp4**: Automatically detected highlight clips

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Detection | YOLO v8 (ultralytics) on NVIDIA GPU |
| Tracking | ByteTrack (custom implementation, no external lib) |
| Camera smoothing | Kalman filter (filterpy) + EMA |
| 360 projection | py360convert (equirectangular to perspective) |
| Video I/O | ffmpeg via subprocess pipes (streaming, no disk frames) |
| Packaging | pyproject.toml, installed as `soccer360` CLI (click) |
| Deployment | Docker (NVIDIA CUDA 12.2 base), docker-compose |
| Testing | pytest |
| Linting | ruff (line-length 100, target py311) |

## Source Layout

```
src/
  pipeline.py     # Orchestrator -- coordinates 8 processing phases
  detector.py     # YOLO ball detection + Field-of-Interest (FoI) filtering
  tracker.py      # ByteTrack ball tracking + ball selection scoring
  camera.py       # Virtual camera path: Kalman + EMA + deadband + dynamic FOV
  reframer.py     # 360->perspective rendering with 12 parallel segment workers
  highlights.py   # Heuristic highlight detection (speed, direction, goal-box)
  exporter.py     # Output organization, metadata, artifact preservation
  watcher.py      # Watchdog daemon monitoring ingest folder
  trainer.py      # YOLO fine-tuning + TensorRT export + hard frame export
  cli.py          # Click CLI entry point
  utils.py        # FFmpeg streaming I/O, angle math, config loading, JSONL I/O
```

## Pipeline Flow

```
Input 360 video
  |
  v
[1] Detector.run_streaming()     -- GPU: ffmpeg pipe -> YOLO batches -> detections
  |                                  Includes FoI filtering (yaw/pitch gating)
  v  detections.jsonl
[2] Tracker.run()                -- CPU: ByteTrack association -> ball positions
  |
  v  tracks.json
[3] CameraPathGenerator.run()    -- CPU: Kalman + EMA -> per-frame yaw/pitch/FOV
  |
  v  camera_path.json
[4] Reframer.run()               -- CPU: py360convert e2p, 12 parallel workers
  |
  v  broadcast.mp4
[5] Reframer.run_tactical()      -- CPU: fixed wide-angle view, parallel
  |
  v  tactical_wide.mp4
[6] HighlightDetector.run()      -- CPU: heuristic events -> highlight clips
  |
  v  highlight_*.mp4
[7] Exporter.run()               -- I/O: organize outputs, write metadata
  |
[8] Cleanup scratch
```

## Data Formats

**detections.jsonl** (one JSON object per line):
```json
{"frame": 0, "bbox": [x1, y1, x2, y2], "confidence": 0.87, "class": 0}
```

**tracks.json** (array of per-frame objects):
```json
[{"frame": 0, "ball": {"x": 960, "y": 480, "bbox": [...], "confidence": 0.85, "track_id": 1}}]
```

**camera_path.json** (array of per-frame camera params):
```json
[{"frame": 0, "yaw": 12.3, "pitch": -5.1, "fov": 90.0}]
```

**foi_meta.json** (runtime FoI metadata):
```json
{"enabled": true, "center_mode": "fixed", "effective_center_yaw_deg": 0.0, "yaw_window_deg": 200, "pitch_min_deg": -45, "pitch_max_deg": 20, "sample_count": 0, "fallback": false}
```

## Critical Math

These formulas are used throughout the codebase. Getting them wrong breaks the pipeline:

**Equirectangular pixel to spherical angles** (`src/utils.py`):
```python
yaw = (x / width) * 360.0 - 180.0      # [-180, +180]
pitch = 90.0 - (y / height) * 180.0     # [+90 top, -90 bottom]
```

**Angle wrapping to (-180, 180]** (`src/utils.py`):
```python
a = a % 360.0
if a > 180.0:
    a -= 360.0
```

**Vertical FOV from horizontal FOV** (`src/reframer.py`):
```python
fov_v = degrees(2 * atan(tan(radians(fov_h / 2)) * (out_h / out_w)))
```
This is the correct tangent-based formula. Do NOT use `fov_h * (h/w)` -- that is a linear approximation that gives wrong results (50.6 vs 58.7 for 16:9 at 90 hFOV).

**FoI auto-center** (`src/detector.py`):
- Build 72-bin histogram (5 degrees each) of detection yaws from first 30s
- Find peak bin
- Compute circular mean (atan2 of sin/cos sum) from peak +/- 1 neighbor bin

## Configuration

All tunable parameters live in `configs/pipeline.yaml`. When adding new parameters:
1. Add to `configs/pipeline.yaml` with a comment
2. Read in the relevant module's `__init__` with a sensible default
3. Add to `tests/conftest.py` test config fixture

## Testing

Tests run inside Docker (no local Python environment):
```bash
docker compose run --rm worker pytest tests/ -v
```

For quick local syntax validation:
```bash
python3 -c "import ast; ast.parse(open('src/detector.py').read())"
```

Test config in `tests/conftest.py` has FoI disabled and uses small resolutions to keep tests fast.

## Common Pitfalls

1. **No local venv**: All execution happens in Docker. `pip install` / `pytest` won't work locally.
2. **Angle convention**: The codebase uses `(-180, 180]` everywhere. `wrap_angle_deg` in utils.py and `wrap_angle` in camera.py both follow this. Don't mix conventions.
3. **Pixel space**: Tracker thresholds (`max_speed_px_per_frame`, `max_displacement_px`, bbox area limits) are all in `detector.resolution` pixel space (default 1920x960), not source video resolution.
4. **Streaming design**: Frames flow through ffmpeg pipes, never written to disk. This is intentional for performance. Don't add intermediate frame storage.
5. **Reframer parallelism**: `_render_segment` and `_render_tactical_segment` are module-level functions (not methods) because ProcessPoolExecutor needs pickleable callables.
6. **Segment overlap**: Adjacent parallel segments overlap by `overlap_sec` (0.5s) for codec warmup. The first segment starts at frame 0 with no overlap.
