# Soccer360 - Gemini Context

## Project Summary

Soccer360 is a Python 3.11 pipeline that converts raw equirectangular 360 soccer match video into broadcast-quality auto-follow footage, a tactical wide-angle view, and highlight clips. It runs in Docker on an NVIDIA GPU server (Tesla P40, Ubuntu 22.04).

## How It Works

The pipeline has 8 sequential phases:

1. **Ball Detection** (`src/detector.py`): Streams video frames via ffmpeg pipe into YOLO batch inference on GPU. Applies Field-of-Interest (FoI) filtering to reject balls detected on adjacent fields. Outputs `detections.jsonl`.

2. **Ball Tracking** (`src/tracker.py`): Runs ByteTrack two-stage association (high-confidence first, then low-confidence) with Kalman box filters over the detections. Selects the most likely ball per frame using a multi-factor score (60% confidence + 40% motion continuity). Outputs `tracks.json`.

3. **Camera Path** (`src/camera.py`): Converts ball pixel positions to spherical angles (yaw/pitch), then smooths with Kalman filter (4-state constant-velocity model), EMA, deadband, velocity threshold, and pan speed clamping. Outputs `camera_path.json` with per-frame yaw, pitch, FOV.

4. **Broadcast Reframing** (`src/reframer.py`): Reads the 360 video again via ffmpeg pipe and uses `py360convert` e2p to extract a perspective crop for each frame based on camera_path.json. Uses 12 parallel workers processing video segments with overlap for codec warmup.

5. **Tactical View** (`src/reframer.py`): Same parallel strategy but with fixed camera at yaw=0, pitch=-5, FOV=120.

6. **Highlights** (`src/highlights.py`): Detects events by ball speed (95th percentile), direction changes (>90 degrees), and goal-box entry. Clusters events and exports clips.

7. **Export** (`src/exporter.py`): Moves outputs to final directories, writes metadata, preserves intermediate artifacts.

8. **Cleanup**: Removes scratch working directory.

## Source Files

| File | Lines | Role |
|------|-------|------|
| `src/pipeline.py` | 128 | Master orchestrator |
| `src/detector.py` | ~480 | YOLO detection + FoI filtering |
| `src/tracker.py` | ~400 | ByteTrack tracking + ball selection |
| `src/camera.py` | ~335 | Camera smoothing (Kalman + EMA) |
| `src/reframer.py` | ~340 | 360-to-perspective rendering |
| `src/highlights.py` | ~250 | Highlight detection |
| `src/exporter.py` | ~150 | Output finalization |
| `src/watcher.py` | ~200 | Watchdog file daemon |
| `src/trainer.py` | ~100 | YOLO fine-tuning |
| `src/cli.py` | 85 | Click CLI |
| `src/utils.py` | ~350 | FFmpeg I/O, angle math, helpers |

## Key Configuration (`configs/pipeline.yaml`)

The config has sections for: `paths`, `model`, `detector`, `field_of_interest`, `tracker`, `camera`, `reframer`, `highlights`, `exporter`, `watcher`, `logging`.

Important parameters:
- `detector.resolution: [1920, 960]` -- detection coordinate space
- `field_of_interest.center_mode: fixed|auto` -- FoI yaw filtering mode
- `camera.ema_alpha: 0.15` -- camera smoothing (lower = smoother)
- `camera.deadband_deg: 0.5` -- suppress micro-oscillation
- `reframer.num_workers: 12` -- parallel rendering workers
- `reframer.overlap_sec: 0.5` -- segment overlap for clean cuts

## Critical Implementation Details

### Angle Math
- Equirect pixel to angle: `yaw = (x/W)*360-180`, `pitch = 90-(y/H)*180`
- All angles wrap to `(-180, 180]` using modulo + conditional subtract
- Vertical FOV uses tangent formula: `fov_v = 2*atan(tan(fov_h/2) * h/w)` (not linear)

### Streaming Architecture
- Video frames are NEVER written to disk as intermediate files
- All video I/O goes through ffmpeg subprocess pipes (stdin/stdout)
- `FFmpegFrameReader` yields numpy arrays, `FFmpegFrameWriter` accepts them

### Parallel Rendering
- Video split into segments, processed by ProcessPoolExecutor
- Render functions are module-level (not class methods) for pickle compatibility
- Adjacent segments overlap by 0.5s for H.264 codec warmup

### Field-of-Interest
- Rejects detections outside a yaw/pitch window
- Auto mode: builds 5-degree yaw histogram from first 30s, locks to peak cluster
- Writes `foi_meta.json` with runtime-computed center

### Tracker Pixel Space
- All tracker thresholds (max_speed, max_displacement, bbox area) are in `detector.resolution` coordinate space, not source video resolution

## Testing

```bash
# All tests (inside Docker)
docker compose run --rm worker pytest tests/ -v

# Single module
docker compose run --rm worker pytest tests/test_detector.py -v
```

No local Python environment exists. Tests must run in Docker.

## Dependencies

ultralytics (YOLO), py360convert, filterpy, scipy, watchdog, click, opencv-python-headless, numpy, pyyaml. Video I/O via system ffmpeg.
