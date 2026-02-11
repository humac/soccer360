# Soccer360

Automated 360 soccer video processing pipeline. Drop a 360 match recording into the ingest folder, and the system produces a broadcast-style auto-follow video, tactical wide-angle view, and highlight clips -- no manual editing required.

## Hardware Requirements

- Bare metal Ubuntu 22.04
- Dual Xeon (or equivalent multi-core CPU)
- 256GB RAM
- NVIDIA Tesla P40 (24GB VRAM)
- 512GB NVMe mounted at `/scratch`
- 4TB SSD mounted at `/tank`

## Quick Start

```bash
# 1. Complete server setup (see SERVER_SETUP.md)

# 2. Clone and install
git clone <repo-url> ~/soccer360
cd ~/soccer360
bash scripts/install.sh

# 3. Start the watcher daemon
docker compose up -d worker

# 4. Drop a 360 video into ingest
cp match.mp4 /tank/ingest/

# 5. Check progress
docker compose logs -f worker

# 6. Find outputs in /tank/processed/<match_name>/
```

## Architecture

Two-pass streaming pipeline designed to process 1-hour 5.7K matches in under 90 minutes:

```
360 video --> Detection (GPU) --> FoI Filter --> Tracking --> Camera Path --> Reframing --> Export
```

**Pass 1 (GPU):** Frames streamed via ffmpeg pipe -> YOLO batch inference -> Field-of-Interest filtering -> detections JSONL. Frames never touch disk.

**Pass 2 (CPU, parallel):** Frames streamed again -> py360convert e2p with per-frame camera angles -> encoded via ffmpeg. 12 parallel workers with segment overlap for clean cuts.

### Processing Phases

| Phase | Operation | Hardware |
|-------|-----------|----------|
| 1 | Ball detection (YOLO) + FoI filtering | GPU (P40) |
| 2 | Ball tracking (ByteTrack) | CPU |
| 3 | Camera path (Kalman filter + EMA) | CPU |
| 4 | Broadcast reframing (py360convert) | CPU (12 workers) |
| 5 | Tactical wide view | CPU (parallel) |
| 6 | Highlight detection & export | CPU |
| 7 | Output organization | I/O |
| 8 | Scratch cleanup | I/O |

## Field-of-Interest (FoI) Filtering

When the 360 camera sits between adjacent soccer fields, YOLO detects balls on both. The FoI filter removes detections outside a configurable yaw/pitch window so only the target game reaches the tracker.

**Modes:**
- **fixed** (default): Use `center_yaw_deg` directly. Camera front lens faces the target field, so `center_yaw_deg: 0` with `yaw_window_deg: 200` covers the front hemisphere.
- **auto**: Analyzes the first N seconds of detections, builds a 5-degree yaw histogram, finds the dominant cluster via peak detection + circular mean, and locks the center for the rest of the run.

**Runtime metadata:** `foi_meta.json` is written to the output directory with the effective center yaw, sample count, and whether fallback was used.

## CLI Commands

```bash
# Watch folder daemon (runs continuously)
soccer360 watch

# Process a single file
soccer360 process /path/to/video.mp4
soccer360 process /path/to/video.mp4 --no-cleanup

# Train/fine-tune ball detection model
soccer360 train --epochs 50 --data /tank/labeling/dataset.yaml

# Export low-confidence frames for labeling
soccer360 export-hard-frames /path/to/video.mp4 /path/to/detections.jsonl --threshold 0.3
```

All commands accept `--config` / `-c` to specify a custom config file (default: `configs/pipeline.yaml`).

## Storage Layout

```
/scratch/                   Fast NVMe, active processing only (auto-cleaned)
/tank/
  +-- ingest/               Drop raw 360 videos here
  +-- processed/            Final outputs (broadcast + tactical + metadata)
  |   +-- <match_name>/
  |       +-- broadcast.mp4
  |       +-- tactical_wide.mp4
  |       +-- camera_path.json
  |       +-- detections.jsonl
  |       +-- tracks.json
  |       +-- foi_meta.json
  |       +-- metadata.json
  +-- highlights/           Highlight clips
  |   +-- <match_name>/
  +-- models/               Trained YOLO models (versioned)
  +-- labeling/             Hard frames + labels for training
  +-- archive_raw/          Optional raw file archive
  +-- logs/                 Pipeline logs
```

## Configuration

All parameters are in `configs/pipeline.yaml`:

- **paths** -- ingest, scratch, processed, models, etc.
- **model** -- YOLO model path, TensorRT toggle, inference backend (`fp32` or `tensorrt_int8`)
- **detector** -- batch size, detection resolution, confidence threshold, frame skipping, tiling
- **field_of_interest** -- enable/disable, center mode (fixed/auto), yaw window, pitch range, auto-center sampling
- **tracker** -- ByteTrack thresholds, track buffer, ball selection sanity checks (detector.resolution pixel space)
- **camera** -- pan speed limits, FOV range, Kalman filter noise, deadband, velocity threshold, ball-lost behavior
- **reframer** -- output resolution, source downscale, worker count, segment overlap, tactical view (FOV 120)
- **highlights** -- speed percentile, direction change threshold, goal-box regions, clip margins
- **exporter** -- codec, CRF quality, encoder (cpu/nvenc), raw file handling
- **watcher** -- file extensions, staging suffix ignore list, stability checks (5x10s)

## Camera Smoothing

The virtual camera uses a multi-stage smoothing pipeline:

1. **Kalman filter** (4-state: yaw, pitch, velocity) -- handles noise and predicts through ball occlusions
2. **EMA post-smoothing** -- removes residual jitter
3. **Deadband** -- ignores movements below configurable angular threshold to prevent micro-oscillation
4. **Velocity threshold** -- camera doesn't react to tiny ball movements
5. **Pan speed clamping** -- enforces broadcast-quality maximum angular velocity (60 deg/s normal, 120 deg/s fast action)
6. **Dynamic FOV** -- widens during fast ball movement, tightens during slow play, immediately widens when ball is lost

Ball-lost fallback:
- Immediately: FOV widens to maximum (configurable)
- Frames 1-30: coast on predicted velocity
- Frames 31-90: velocity decays, camera slows
- Frames 91+: slowly drift toward field center

## Model Training Workflow

```bash
# 1. Process a match to generate detections
soccer360 process /tank/ingest/match.mp4 --no-cleanup

# 2. Export hard frames (low confidence + lost ball)
soccer360 export-hard-frames match.mp4 detections.jsonl --threshold 0.3

# 3. Label in Label Studio
docker compose up -d labelstudio
# Open http://localhost:8080, import images, annotate ball bounding boxes

# 4. Export labels in YOLO format, create dataset.yaml

# 5. Train
bash scripts/train.sh 50 /tank/labeling/dataset.yaml

# 6. Update config to use new model
# Edit configs/pipeline.yaml -> model.path
```

## Docker Services

```bash
docker compose up -d worker        # Start processing daemon
docker compose up -d labelstudio   # Start Label Studio (port 8080)
docker compose logs -f worker      # Follow logs
docker compose down                # Stop everything
```

## Tesla P40 Notes

The P40 is a Pascal GP102 GPU with 24GB VRAM. It supports FP16 arithmetic (no Tensor Cores) and has an NVENC hardware encoder.

**Baseline** (default): FP32 inference + CPU encoding. This is the correctness-first path.

**Optimizations** (after baseline is validated):

1. **TensorRT INT8**: Export model and set `model.backend: tensorrt_int8` in config. 47 TOPS INT8 vs 12 TFLOPS FP32.
2. **NVENC encoding**: Set `exporter.encoder: nvenc` to use hardware encoding instead of CPU libx264.
3. **Frame skipping**: Set `detector.process_every_n_frames: 2` to halve GPU detection load (positions interpolated).
4. **Source downscale**: Set `reframer.source_downscale: [3840, 1920]` to downscale before reframing.

## Project Structure

```
src/
  cli.py          Click CLI entry point
  watcher.py      Watchdog folder daemon (stability checks, staging suffix ignore)
  pipeline.py     Orchestrator (coordinates all 8 processing phases)
  detector.py     YOLO streaming batch detection + Field-of-Interest filtering
  tracker.py      ByteTrack ball tracking (two-stage association, Kalman box filter)
  camera.py       Camera path generation (Kalman + EMA + deadband + dynamic FOV)
  reframer.py     360-to-perspective rendering (parallel segments, overlap warmup)
  highlights.py   Heuristic highlight detection (speed, direction, goal-box events)
  exporter.py     Output organization + metadata + artifact preservation
  trainer.py      YOLO fine-tuning + TensorRT export + hard frame export
  utils.py        FFmpeg streaming I/O, config, equirectangular angle helpers

configs/
  pipeline.yaml       Main processing configuration
  model_config.yaml   YOLO training configuration

scripts/
  install.sh      Server installation (directories, Docker build, verification)
  train.sh        Model training wrapper

tests/
  conftest.py         Shared pytest fixtures
  test_pipeline.py    End-to-end pipeline tests
  test_detector.py    Detection + FoI filter + NMS tests
  test_tracker.py     ByteTrack association + ball selection tests
  test_camera.py      Camera path smoothing + angle conversion tests
  test_reframer.py    Vertical FOV math + e2p integration tests
  test_highlights.py  Highlight detection tests
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| ultralytics | YOLO ball detection |
| py360convert | Equirectangular-to-perspective projection |
| filterpy | Kalman filtering (tracking + camera smoothing) |
| scipy | Linear sum assignment (ByteTrack matching) |
| watchdog | File system event monitoring |
| click | CLI framework |
| opencv-python-headless | Image processing |
| ffmpeg (system) | Video decode/encode via streaming pipes |

## Testing

```bash
# Run all tests inside Docker
docker compose run --rm worker pytest tests/ -v

# Run specific test module
docker compose run --rm worker pytest tests/test_detector.py -v

# Run with coverage
docker compose run --rm worker pytest tests/ --cov=src --cov-report=term-missing
```
