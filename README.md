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

```text
360 video --> Detection (GPU) --> FoI Filter --> Tracking --> Player Cluster --> Camera Path --> Reframing --> Export
```

**Pass 1 (GPU):** Frames streamed via ffmpeg pipe -> YOLO batch inference (ball + players) -> Field-of-Interest filtering -> detections JSONL. Frames never touch disk.

**Pass 2 (CPU, parallel):** Frames streamed again -> py360convert e2p with per-frame camera angles -> encoded via ffmpeg. 12 parallel workers with segment overlap for clean cuts.

### Processing Phases

|Phase|Operation|Hardware|
|---|---|---|
|1|Ball + player detection (YOLO) + FoI filtering|GPU (P40)|
|2|Ball tracking (BallStabilizer / ByteTrack)|CPU|
|2.5|Hard frame export (active learning)|I/O|
|2.7|Player cluster (center of play)|CPU|
|3|Camera path (Kalman filter + EMA + hybrid blend)|CPU|
|4|Broadcast reframing (py360convert)|CPU (12 workers)|
|5|Tactical wide view|CPU (parallel)|
|6|Highlight detection & export|CPU|
|7|Output organization|I/O|
|8|Scratch cleanup|I/O|

### V1 Bootstrap Detection

The V1 pipeline uses YOLO26l (`yolo26l.pt`, the latest Ultralytics YOLO generation — NMS-free, 43% faster CPU inference than YOLO11) detecting both sports ball (class 32) and person (class 0) with conservative filtering and temporal stabilization. Person detections feed the center-of-play module; ball detections feed the tracker. This enables a train-then-upgrade cycle:

1. **Detect** -- YOLO detects balls and players with class filter + y-range filter + best-per-frame selection (best ball per frame; all person detections passed through)
2. **Stabilize** -- BallStabilizer applies persistence gate (require N of M frames), jump/speed rejection, and EMA smoothing
3. **Export hard frames** -- ActiveLearningExporter flags low-confidence detections, lost ball runs, and jump rejections for labeling
4. **Label** -- Annotate exported frames in Label Studio
5. **Train** -- Fine-tune with `soccer360 train`, producing `ball_best.pt`
6. **Upgrade** -- Drop `ball_best.pt` in `/tank/models/`; next run auto-uses it

**Temporal stabilization** prevents false positives from reaching the camera path:

- **Persistence gate**: Requires N detections within a sliding window of M frames before accepting
- **Jump rejection**: Rejects detections that teleport beyond `max_jump_px` pixels
- **Speed rejection**: Rejects detections moving faster than `max_speed_px_per_s`
- **EMA smoothing**: Exponential moving average on accepted ball positions

**Active learning triggers** (three types of hard frames exported):

- **Low confidence**: Detection confidence in configurable range `[low_conf_min, low_conf_max]`
- **Lost ball runs**: ONE representative frame per streak of `lost_run_frames` consecutive lost frames
- **Jump rejections**: Tracking events where distance exceeds `jump_trigger_px`

### Model Resolution

**V1 mode** (when `detection` section is present) resolves with stable precedence:

1. `detector.model_path` (canonical override)
2. `detection.path` (legacy fallback)
3. `default` resolver path (`/tank/models/ball_best.pt` when present, else baked `/app/yolov8s.pt`)

Default behavior is unchanged unless `detector.model_path` is set, or a legacy
`detection.path` override is already in use.

Notes:

- `detector.model_path` set to a non-default path is treated as explicit override.
- Explicit non-default `detector.model_path` must exist and be a file; otherwise V1 model resolution fails fast with `RuntimeError`.
- `detector.model_path: /app/yolov8s.pt` keeps normal default/fine-tuned behavior.
- Current contract is `.pt` weights only (ONNX/TRT model-path selection is out of scope here).
- Runtime logs one line per job: `Model resolved: <path> (source=<source>)`.
- Source enum is stable: `detector.model_path`, `detection.path`, `default`.

### Roboflow Model Placement (Default Compose Runtime)

- Canonical container path: `/app/models/roboflow/football_players_v1.pt`
- In the default `docker compose` worker runtime, `/app/models` is bind-mounted from host `/tank/models`.
- Host placement for Roboflow weights in this runtime: `/tank/models/roboflow/football_players_v1.pt`

Example config override:

```yaml
detector:
  model_path: /app/models/roboflow/football_players_v1.pt
```

**Legacy mode** (no `detection` section):

1. `/tank/models/ball_best.pt` (fine-tuned model)
2. Config `model.path` (`/app/models/ball_best.pt`)
3. `/app/models/ball_base.pt` (base model, auto-copied to tank)
4. No model found -- NO_DETECT mode

### Model Fallback (NO_DETECT mode)

If no ball detection model is available, the pipeline runs in **NO_DETECT mode**:

- Skips phases 1, 2, 2.5, 2.7, and 6 (detection, tracking, hard frames, player cluster, highlights)
- Generates a static camera path at field center with default FOV
- Still produces `broadcast.mp4` (fixed framing) and `tactical_wide.mp4`
- `metadata.json` includes `"mode": "no_detect"` to indicate degraded output

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

# Start monitoring dashboard (port 8088)
soccer360 dashboard
soccer360 dashboard --port 9000 --host 127.0.0.1

# Train/fine-tune ball detection model
soccer360 train --epochs 50 --data /tank/labeling/dataset.yaml

# Export low-confidence frames for labeling
soccer360 export-hard-frames /path/to/video.mp4 /path/to/detections.jsonl --threshold 0.3
```

All commands accept `--config` / `-c` to specify a custom config file (default: `configs/pipeline.yaml`).

## Storage Layout

```text
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
  |       +-- player_cluster.json
  |       +-- foi_meta.json
  |       +-- hard_frames.json
  |       +-- metadata.json
  +-- highlights/           Highlight clips
  |   +-- <match_name>/
  +-- models/               Trained YOLO models (versioned, timestamped)
  |   +-- ball_best.pt      Active model (auto-picked up by worker)
  |   +-- ball_model_YYYYMMDD_HHMM/  Versioned training runs
  +-- labeling/             Hard frames + labels for training
  |   +-- <match_name>/
  |       +-- frames/       Auto-exported hard frames (JPEG)
  |       +-- labels/       YOLO-format labels (from Label Studio)
  |       +-- hard_frames.json  Manifest with reasons + predicted bboxes
  |       +-- labelstudio/  Label Studio task JSON
  |   +-- dataset/          Built dataset (train/val splits)
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
- **camera** -- pan speed limits, FOV range, FOV EMA smoothing (`fov_ema_alpha`), Kalman filter noise, deadband, velocity threshold, ball-lost behavior
- **reframer** -- output resolution, source downscale, worker count, segment overlap, tactical view (FOV 120)
- **highlights** -- ball detectors (speed, direction, goal-box), cluster detectors (convergence, velocity, goal zone, density), scoring weights, clip margins, max clips
- **exporter** -- codec, CRF quality, encoder (cpu/nvenc), raw file handling
- **watcher** -- file extensions, staging suffix ignore list, stability checks (5x10s), dotfile filtering, persistent processed-state dedupe file
- **ingest** -- post-success archival (archive mode, collision handling, name template)
- **active_learning** -- V1: three-trigger hard frame export (low confidence range, lost ball runs, jump rejections), gating (every_n, max cap)
- **detection** -- V1 bootstrap: YOLO model path (currently `yolo26l.pt`), COCO class filter (`[32, 0]`), confidence/IOU, image size, half precision, device
- **center_of_play** -- hybrid camera tracking: player cluster computation, ball/cluster blend weights, FOV-from-spread, EMA smoothing
- **filters** -- V1: y-range vertical band filter, max jump/speed sanity limits
- **tracking** -- V1: EMA alpha, persistence gate (require_persistence, window size)
- **mode** -- allow_no_model toggle for graceful degradation
- **dashboard** -- monitoring UI: enabled toggle, SQLite db_path, server port

## Camera Smoothing

The virtual camera uses a multi-stage smoothing pipeline:

1. **Hybrid ball + cluster blending** -- when ball detected: 85% ball / 15% player cluster; low confidence: 50/50; ball lost: 100% cluster; both lost: drift to center
2. **Kalman filter** (4-state: yaw, pitch, velocity) -- handles noise and predicts through ball occlusions
3. **EMA post-smoothing** -- removes residual jitter on yaw/pitch
4. **Deadband** -- ignores movements below configurable angular threshold to prevent micro-oscillation
5. **Smooth velocity gain** -- linear ramp from 0.3× to 1.0× based on ball speed (no binary step function)
6. **Smooth pan speed interpolation** -- linearly blends between normal (60 deg/s) and fast (120 deg/s) limits based on ball speed (no binary flicker)
7. **FOV EMA smoothing** -- all FOV changes are smoothed via EMA (`fov_ema_alpha: 0.08`, ~12-frame half-life). Lost→found transitions are gradual, not instant snaps. Spread data is carried forward across cluster gaps to prevent FOV drops on missing frames.

Ball-lost fallback (with center of play enabled):

- **Player cluster available**: camera follows center of play (no drift to field center)
- **No cluster**: coast on predicted velocity (frames 1-30), velocity decays (31-90), drift toward field center (91+)
- **FOV**: targets maximum when lost, but EMA smoothing produces a gradual widen rather than an instant snap; adapts to player spread when cluster available

## Weekly Improvement Loop (Active Learning)

The system improves over time through a simple weekly cycle:

1. **Process games** -- hard frames are exported automatically
2. **Label 5-10 minutes** -- annotate ball bounding boxes in Label Studio
3. **Build dataset + train** -- one command each
4. **Next games are better** -- worker auto-picks up the new model

```bash
# 1. Process games (hard frames exported to /tank/labeling/<match>/frames/)
docker compose up -d worker
cp match1.mp4 match2.mp4 /tank/ingest/

# 2. Import hard frames into Label Studio
bash scripts/labelstudio_import.sh match1
bash scripts/labelstudio_import.sh match2

# 3. Label in Label Studio
#    Open http://<server>:8080
#    Create project, import tasks.json, label ball bounding boxes
#    Export annotations in YOLO format to /tank/labeling/<match>/labels/

# 4. Build dataset from all labeled matches
bash scripts/build_dataset.sh

# 5. Train (50 epochs by default)
bash scripts/train_ball.sh 50

# 6. Next games automatically use /tank/models/ball_best.pt
#    To reprocess a match with the new model:
docker compose run --rm worker soccer360 process /tank/ingest/match.mp4
```

### Ingest Queue and Archival

`/tank/ingest/` is a **queue folder**: drop raw 360 videos here for processing. After a successful run, the original file is automatically archived to `/tank/archive_raw/` (when enabled), keeping the ingest folder clean with only pending jobs.

**Safe ingest:** Use atomic copy to avoid processing partial files:

```bash
# Recommended: copy to .part then rename
cp match.mp4 /tank/ingest/match.mp4.part
mv /tank/ingest/match.mp4.part /tank/ingest/match.mp4
```

The watcher ignores `.part`, `.tmp`, `.uploading` suffixes and hidden files (dotfiles).
Files must have a stable size for 50 seconds (configurable) before processing begins.
The watcher also persists successful ingest fingerprints in
`watcher.processed_state_file` (default
`/tank/processed/.state/watcher_processed_ingest.json`) to prevent reprocessing
loops after daemon restarts. Relative `watcher.processed_state_file` values
resolve under `<paths.processed>/.state/` (persistent storage, not scratch).

The persisted dedupe marker is written when processing completes successfully
(`done` means export completed). Archival bookkeeping failures do not cause
reprocessing loops: even if ingest archival fails, dedupe still marks the run
as processed and `metadata.json` records the archival failure details.

Watcher dedupe state settings:

|Key|Default|Description|
|---|---|---|
|`processed_state_file`|`watcher_processed_ingest.json`|JSON state filename/path; relative values resolve under `<paths.processed>/.state/`|
|`processed_state_max_entries`|`50000`|Retention cap for state entries (latest N records kept, `0` = unlimited)|

Tune `processed_state_max_entries` based on ingest volume and startup latency.
Higher values retain longer dedupe history but increase state file size/load time.
Lower values reduce startup overhead for high-volume deployments.

**Post-success archival** is configured in `configs/pipeline.yaml` under `ingest`:

|Key|Default|Description|
|---|---|---|
|`archive_on_success`|`true`|Enable archival after successful processing|
|`archive_dir`|`/tank/archive_raw`|Destination directory for archived originals|
|`archive_mode`|`move`|`move` (relocate), `copy` (keep original too), or `leave` (disable)|
|`archive_name_template`|`{match}_{job_id}{ext}`|Filename template (`{match}` = input stem, `{job_id}` = pipeline job id, `{ext}` = `.mp4`)|
|`archive_collision`|`suffix`|`suffix` (append `_01`, `_02`), `skip` (leave in ingest), or `overwrite`|

If archival fails (e.g. permissions), the pipeline still succeeds -- processed outputs are preserved, the ingest file stays in place, and `metadata.json` records `ingest_archive_status: "failed"`.
This also applies to `archive_mode: copy` / `leave` and `archive_collision: skip`: persistent watcher dedupe prevents restart loops even when the ingest file remains present.

### Reset Dedupe State (Force Reprocess)

If you intentionally want the watcher to process previously completed ingest files again:

```bash
# 1) Stop watcher
docker compose stop worker

# 2) Reset dedupe state
rm -f /tank/processed/.state/watcher_processed_ingest.json
rm -f /tank/processed/.state/watcher_processed_ingest.json.corrupt.*

# Alternative: point watcher.processed_state_file to a new filename in config

# 3) Restart watcher
docker compose up -d worker
```

If model/config changes and you want reruns, reset dedupe state first.
Automatic signature-based rerun is intentionally deferred.

### Restart Smoke Script

Run the objective restart-loop smoke matrix (copy / leave / deterministic collision=skip):

```bash
scripts/smoke_dedupe_restart.sh /path/to/sample.mp4
```

### Hard Frame Export

During every pipeline run, Phase 2.5 automatically identifies "hard frames" where the model struggled:

- **Low confidence** -- detection confidence below threshold (default 0.3)
- **Lost ball gaps** -- consecutive frames with no ball detected (default >15 frames)
- **Position jumps** -- ball position jumps between frames (default >150px)

Exported to `/tank/labeling/<match>/frames/` with a manifest at `hard_frames.json` containing frame index, timestamp, reason, and predicted bbox/confidence.

Configure in `configs/pipeline.yaml` under `active_learning`.

## Monitoring Dashboard

A web-based monitoring UI on port 8088 provides real-time pipeline visibility and interactive decision handling. No babysitting logs required.

**Features:**

- Real-time pipeline progress (9-phase progress bar with timing)
- GPU utilization gauges (updated via SSE every ~5s)
- CPU and RAM utilization gauges (from `/proc/stat` and `/proc/meminfo`, no psutil dependency)
- Interactive decision prompts (approve/reject with countdown timers)
- Job history with phase-level metrics
- Active learning management (labeling status, dataset build, model training)
- Media player for reviewing processed outputs

**Start the dashboard:**

```bash
# Via Docker Compose (recommended)
docker compose up -d dashboard

# Or directly via CLI
soccer360 dashboard --port 8088
```

Open `http://<server>:8088` in a browser. The dashboard streams events from the pipeline via SSE — no polling, no refresh needed.

**Decision hooks:** When the pipeline reaches a decision point (mode confirmation, post-detection review, hard frame labeling), a notification banner appears with approve/reject buttons and a countdown timer. If no response is given, the pipeline auto-proceeds with the default option.

**Training management:** The Active Learning section shows labeling progress per match, and provides buttons to build a YOLO dataset and trigger model training — all from the browser.

**Configuration** (`configs/pipeline.yaml`):

```yaml
dashboard:
  enabled: true          # Enable event emission from pipeline
  db_path: /tank/data/dashboard.db  # SQLite state store
  port: 8088             # Dashboard server port
```

## Docker Services

```bash
# Start all services at once (worker + dashboard + Label Studio)
docker compose up -d

# Or start individual services
docker compose up -d worker        # Processing daemon (GPU)
docker compose up -d dashboard     # Monitoring dashboard (port 8088)
docker compose up -d labelstudio   # Label Studio (port 8080)

# Useful commands
docker compose logs -f worker      # Follow worker logs
docker compose ps                  # Check service status + health
docker compose down                # Stop everything
```

| Service | Port | URL | Purpose |
| --- | --- | --- | --- |
| worker | — | — | Processing daemon (watches `/tank/ingest/`) |
| dashboard | 8088 | `http://<server>:8088` | Real-time monitoring + training management |
| labelstudio | 8080 | `http://<server>:8080` | Hard frame annotation |

All services include Docker health checks. The `dashboard` service reuses the same `soccer360-worker:local` image as the worker — no separate build needed.

### Verify Worker Image Freshness and Runtime Assets

Root causes this workflow catches:

1. Stale container/image reuse after rebuilds.
2. Compose project/context drift causing a different auto-tagged image.
3. `build` + `image` + pull behavior mismatches.
4. Runtime user (`1000:1000`) permission mismatch on baked assets.
5. Missing passwd/group identity for numeric UID 1000 causing `getpass`/torch runtime crashes.

Worker image policy:

- `docker-compose.yml` sets `image: soccer360-worker:local` + `pull_policy: never`.
- Worker runs as numeric `1000:1000`.
- With both `build` and `image`, compose builds and tags the local result as `soccer360-worker:local`.
- Docker image includes UID/GID 1000 passwd/group compatibility plus `HOME`/`USER`/`LOGNAME` to prevent `getpass.getuser()` failures in torch/ultralytics paths.
- Docker image pins PyTorch for Pascal compatibility: `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `torchaudio==2.4.1+cu121`.
- `pull_policy: never` may not be honored on every Compose version; the verifier script is the source of truth.

**BuildKit required:** The Dockerfile uses BuildKit cache mounts (`RUN --mount=type=cache`).
The verifier script sets `DOCKER_BUILDKIT=1` automatically. For manual builds outside the verifier:

```bash
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose -p soccer360 build worker
```

**Fast dev check** (cached build, does NOT stop running services):

```bash
make verify-container-assets
```

**Pre-merge clean check** (no-cache rebuild, resets compose state):

```bash
make verify-container-assets-clean
```

Both modes:

- Verify `requirements-docker.txt` matches `pyproject.toml` before building (auto docker fallback if host `python3` is missing or lacks `tomllib`/`tomli`).
- Print dependency mismatch details before failing.
- Assert rebuilt image SHA == ephemeral container SHA.
- Resolve model path in-container using the same Python config + resolver logic as runtime (no bash YAML parsing), printing:
  `CONFIG_PATH=...`, `MODEL_PATH=...`, `MODEL_SOURCE=...`
- Resolver stdout contract is strict: only those three `KEY=VALUE` lines are emitted on stdout (in that order).
- Resolver exit codes are deterministic:
  - `11`: config path missing/not a file/not readable
  - `12`: config parse/load failure
  - `13`: resolver import/runtime resolution failure
- Validate selected `MODEL_PATH` is non-empty (`test -s`) and log file size.
- Enforce baked `/app/yolov8s.pt` checks only when `MODEL_PATH=/app/yolov8s.pt`; otherwise skip baked-model checks and validate only the selected path.
- Always validate `/app/.ultralytics` is writable.
- Validate runtime identity resolution via `python -c "import getpass; print(getpass.getuser())"`.
- Print runtime torch/CUDA diagnostics and GPU capability (`nvidia-smi --query-gpu=name,compute_cap`) when available.
- Run CUDA kernel smoke test by default (`GPU_SMOKE=1`); override with `GPU_SMOKE=0` to skip.
- Treat arch-list mismatch as a warning; smoke test is the authoritative kernel gate.

If resolver import/config load fails, verifier still prints the attempted
`CONFIG_PATH`, preserves the resolver exit code, and surfaces resolver stderr before exiting non-zero.

**Environment variable overrides:**

|Variable|Default|Description|
|---|---|---|
|`PROJECT`|`soccer360`|Compose project name|
|`IMAGE_TAG`|`soccer360-worker:local`|Image tag to verify|
|`NO_CACHE`|`0`|Set `1` for `--no-cache` build|
|`RESET`|`0`|Set `1` to run `compose down` before build (applies to both cached and no-cache builds)|
|`SKIP_DEPS_SYNC`|`0`|Set `1` to skip deps sync check|
|`VERBOSE`|`0`|Set `1` to print captured resolver stderr/noise diagnostics when non-empty|

Example with overrides:

```bash
PROJECT=soccer360 IMAGE_TAG=mytag:local bash scripts/verify_container_assets.sh
```

Disable smoke test (not recommended unless debugging):

```bash
GPU_SMOKE=0 make verify-container-assets
```

Because worker service `ENTRYPOINT` is `soccer360`, run Python diagnostics with `--entrypoint python`:

```bash
docker compose run --rm --no-deps --entrypoint python worker -c "
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('arch list:', getattr(torch.cuda, 'get_arch_list', lambda: [])())
print('is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device cap:', torch.cuda.get_device_capability())
"
```

Print resolved model path/source with the same runtime logic used by verifier:

```bash
docker compose run --rm --no-deps --entrypoint python worker -c "
import os
from src.utils import load_config
from src.detector import resolve_v1_model_path_and_source

config_path = (os.getenv('SOCCER360_CONFIG') or '/app/configs/pipeline.yaml')
cfg = load_config(config_path)
model_path, model_source = resolve_v1_model_path_and_source(
    cfg, models_dir=cfg.get('paths', {}).get('models', '/app/models')
)
print(f'CONFIG_PATH={config_path}')
print(f'MODEL_PATH={model_path}')
print(f'MODEL_SOURCE={model_source}')
"
```

### Roboflow Quick Validation

```bash
# 1) Place Roboflow weights on host (default compose runtime mount)
mkdir -p /tank/models/roboflow
cp /path/to/best.pt /tank/models/roboflow/football_players_v1.pt

# 2) Set detector.model_path in configs/pipeline.yaml
# detector:
#   model_path: /app/models/roboflow/football_players_v1.pt

# 3) Confirm model class names from the worker image
docker compose run --rm --no-deps --entrypoint python worker -c "
from ultralytics import YOLO
m = YOLO('/app/models/roboflow/football_players_v1.pt')
print(m.names)
"

# 4) Validate resolver/verifier selection
make verify-container-assets
```

Expected verifier lines with Roboflow selected:

- `resolved_model_path=/app/models/roboflow/football_players_v1.pt`
- `resolved_model_source=detector.model_path`
- `resolved_model_size_bytes=<non-zero>`
- `Resolved model is not /app/yolov8s.pt; skipping baked yolov8s.pt checks`

**Dependency sync check** (standalone):

`requirements-docker.txt` mirrors `pyproject.toml` dependencies for Docker layer caching. The verifier runs this automatically; to run it standalone:

```bash
make check-deps-sync
```

If host `python3` cannot import `tomllib`/`tomli`, the verifier falls back to running the check in `python:3.11-slim` (requires Docker CLI + daemon).

## Tesla P40 Notes

The P40 is a Pascal GP102 GPU with 24GB VRAM. It supports FP16 arithmetic (no Tensor Cores) and has an NVENC hardware encoder.

**Compatibility requirement:** P40 is compute capability `sm_61`. Recent torch builds (for example CUDA 12.8-era wheels) may omit Pascal kernels and fail with `no kernel image is available for execution on the device`.
This image pins a Pascal-compatible stack from cu121 (`torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `torchaudio==2.4.1+cu121`).

**Baseline** (default): FP32 inference + CPU encoding. This is the correctness-first path.

**Optimizations** (after baseline is validated):

1. **TensorRT INT8**: Export model and set `model.backend: tensorrt_int8` in config. 47 TOPS INT8 vs 12 TFLOPS FP32.
2. **NVENC encoding**: Set `exporter.encoder: nvenc` to use hardware encoding instead of CPU libx264.
3. **Frame skipping**: Set `detector.process_every_n_frames: 2` to halve GPU detection load (positions interpolated).
4. **Source downscale**: Set `reframer.source_downscale: [3840, 1920]` to downscale before reframing.

## Project Structure

```text
src/
  cli.py          Click CLI entry point (watch, process, train, dashboard)
  watcher.py      Watchdog folder daemon (stability checks, dotfile filtering, EventBus)
  pipeline.py     Orchestrator (phases, event bus, decision hooks, NO_DETECT mode)
  detector.py     YOLO streaming batch detection + FoI filtering + model resolution
  tracker.py      ByteTrack ball tracking (two-stage association, Kalman box filter)
  player_cluster.py  Player cluster computation (center of play, trimmed mean, EMA)
  camera.py       Camera path generation (hybrid blend + Kalman + EMA + deadband + FOV EMA smoothing)
  reframer.py     360-to-perspective rendering (parallel segments, overlap warmup)
  highlights.py   Heuristic highlight detection (ball + cluster signals, scoring/ranking)
  exporter.py     Output organization + metadata + artifact preservation
  hard_frames.py       Automatic hard-frame export for active learning (legacy)
  active_learning.py   V1 active learning export (three-trigger identification)
  trainer.py           YOLO fine-tuning + TensorRT export
  utils.py             FFmpeg streaming I/O, config, equirectangular angle helpers
  events.py            EventStore (SQLite WAL) + EventBus + decision queue
  dashboard.py         FastAPI monitoring dashboard + REST API + SSE + training mgmt
  metrics.py           PhaseTimer (per-phase timing) + GPU/CPU/RAM utilization snapshots
  static/
    index.html         Single-page dashboard UI (vanilla JS/CSS, SSE)

configs/
  pipeline.yaml       Main processing configuration
  model_config.yaml   YOLO training configuration

scripts/
  install.sh              Server installation (directories, Docker build, verification)
  train.sh                Legacy model training wrapper
  train_ball.sh           Active learning training (timestamp-versioned)
  build_dataset.sh        Build YOLO dataset from labeled matches
  labelstudio_import.sh   Generate Label Studio task JSON from hard frames

tests/
  conftest.py              Shared pytest fixtures
  test_pipeline.py         End-to-end pipeline tests
  test_detector.py         Detection + FoI filter + NMS tests
  test_tracker.py          ByteTrack association + ball selection tests
  test_player_cluster.py   Player cluster computation + EMA smoothing tests
  test_camera.py           Camera path smoothing + hybrid blend + angle conversion tests
  test_reframer.py         Vertical FOV math + e2p integration tests
  test_highlights.py       Highlight detection tests
  test_hard_frames.py           Hard frame identification + export tests
  test_bootstrap_detection.py  V1 model resolution + y-range + best-per-frame tests
  test_ball_stabilizer.py      V1 persistence gate + jump/speed rejection + EMA tests
  test_active_learning.py      V1 three-trigger export + gating tests
  test_model_resolution.py     Model resolution + NO_DETECT mode tests
  test_watcher.py          Watcher ingest handling + safety tests
  test_exporter.py         Output organization tests
  test_events.py           EventStore + EventBus + decision queue tests
  test_dashboard.py        Dashboard API + training endpoint tests
  test_metrics.py          PhaseTimer + GPU snapshot tests
```

## Key Dependencies

| Package | Purpose |
| --- | --- |
| ultralytics | YOLO ball detection |
| py360convert | Equirectangular-to-perspective projection |
| filterpy | Kalman filtering (tracking + camera smoothing) |
| scipy | Linear sum assignment (ByteTrack matching) |
| watchdog | File system event monitoring |
| click | CLI framework |
| opencv-python-headless | Image processing |
| ffmpeg (system) | Video decode/encode via streaming pipes |
| fastapi | Monitoring dashboard REST API |
| uvicorn | ASGI server for dashboard |
| sse-starlette | Server-Sent Events for real-time streaming |

## Testing

```bash
# Run all tests inside Docker
docker compose run --rm worker pytest tests/ -v

# Run specific test module
docker compose run --rm worker pytest tests/test_detector.py -v

# Run with coverage
docker compose run --rm worker pytest tests/ --cov=src --cov-report=term-missing
```
