# Soccer360 - Functional Discovery Notes

> Internal working document. Not a deliverable.

## Application Type

Soccer360 is a **CLI-first video processing pipeline with a web monitoring/operations dashboard**. It processes equirectangular 360-degree soccer match recordings into broadcast-quality outputs. Deployed as Docker services with a persistent file-watching daemon plus a FastAPI dashboard for monitoring, training, staging import, and reprocessing flows.

## User Roles

| Role | Description | Interface |
| ---- | ----------- | --------- |
| **Operator** | Day-to-day user: uploads/stages videos, monitors processing, retrieves outputs, labels hard frames, manages matches | Dashboard (port 8088), Label Studio (port 8080) |
| **Administrator** | Sets up server, configures pipeline, manages Docker, handles GPU/model lifecycle, troubleshoots | Shell access, Docker Compose, YAML config, scripts |

## UI Pages

| Page | URL | Description |
|------|-----|-------------|
| Main Dashboard | `/` (port 8088) | SPA with 4 workspaces |
| Match Playback | `/match/{name}` | Per-match video player with outputs, lifecycle, lineage |
| Detection Settings | `/settings/detection` | Read-only config viewer |

## Dashboard Workspaces

### Overview
- Live Pipeline status (current job, 9-phase progress strip)
- Hero metrics (queue depth, attention count, dataset state, training state)
- Matches Needing Attention (labeling/training pipeline matches only)
- System Signals (GPU util, GPU memory, CPU, RAM, temperature)
- Recent Job History

### Matches
- Match Library table with search + 6 status filters (All, Processing, Staged, Labeling, Ready To Train, Processed)
- Columns: Match, Status, Outputs, Labeling, Model Lineage, Actions
- Row click opens inspect drawer (lifecycle, lineage, outputs, reset preview)
- Next Action button (context-sensitive: Import, Generate Tasks, Open Label Studio, View Playback, etc.)
- Delete button per row (staged → delete file, processed → reset family with confirmation)
- Busy spinner on async actions

### Labeling & Training
- Per-Match Labeling Flow cards with 3-step stepper: Frames → Tasks → Labels
- Dataset State panel (global: total frames, labeled count, dataset state, Build Dataset button)
- Training Console (base model selector, epochs, output name, Train Model button, live status/log)

### Models & Files
- Staging Inbox (resumable upload with progress, search, bulk send to ingest)
- Current Ingest Models (ball/player model selection dropdowns)
- Model Registry (list with role tags, delete, size/usage info)
- Full Job History table

## Core Operator Workflows

1. **Upload & Ingest**: Upload video to staging → Send to ingest → Watcher auto-processes
2. **Pipeline Monitoring**: Real-time SSE phase progress + GPU/CPU gauges + decision prompts
3. **Match Review**: Match Library → filter/search → inspect drawer → playback page
4. **Labeling**: Hard frames auto-exported → Generate Tasks → Label in Label Studio → Upload labels
5. **Training**: Build Dataset (combines all matches) → Configure & Train → Model in registry
6. **Match Management**: Delete staged files or reset processed families (restore to staging)

## Data Model

### Persistence
- **SQLite EventStore** (`/tank/data/dashboard.db`): jobs, phase_events, metrics_snapshots, decisions
- **File System**: all video/frame/label/model artifacts in `/tank/` directory tree
- **No user accounts or authentication** — single-operator system

### Key Directories
```
/tank/ingest/          → queue folder (watcher picks up files)
/tank/stagging/        → manual staging for UI-managed uploads/imports
/scratch/work/         → NVMe temp space (auto-cleaned per job)
/tank/processed/       → final outputs (broadcast.mp4, tactical_wide.mp4, metadata.json)
/tank/highlights/      → highlight clips per match
/tank/models/          → YOLO weights (.pt files)
/tank/labeling/        → hard frames, labels, dataset
/tank/archive_raw/     → archived original recordings
/tank/logs/            → pipeline logs
```

### Processing Artifacts (per match)
- `detections.jsonl` — per-frame YOLO detections (ball class 32 + person class 0)
- `tracks.json` — tracked/stabilized ball positions
- `player_cluster.json` — per-frame player cluster centroid and spread
- `camera_path.json` — per-frame virtual camera (yaw, pitch, FOV)
- `hard_frames.json` — hard-frame candidates with trigger reasons
- `metadata.json` — pipeline execution record

### Runtime Modes
1. **V1 Detection** (`detection` section present) — YOLO + BallStabilizer + active learning + center-of-play
2. **Legacy** (no `detection` section) — YOLO + ByteTrack + center-of-play
3. **NO_DETECT** (model unavailable) — static camera, broadcast + tactical only

## API Surface

36 endpoints total (unauthenticated): status, jobs, decisions, training/labeling (including TrackNetV3 training), staging/upload, media/video streaming, SSE event stream.

New TrackNetV3 training endpoints (Phase 3):
- `POST /api/training/build-tracknet-dataset` — converts YOLO labels to Gaussian heatmaps for TrackNetV3
- `POST /api/training/train-tracknet` — launches TrackNetV3 training in background thread
- `GET /api/training/tracknet-status` — returns TrackNetV3 training state

## Configuration

Single file: `configs/pipeline.yaml` with 15+ sections: paths, detector, detection, camera, center_of_play, tracker, highlights, active_learning, reframer, exporter, ingest, watcher, dashboard, logging.

## Recent Feature Additions (Pipeline Upgrade Phases)

### Phase 1: Cinematic Camera Controller

New camera smoothing features (all default to disabled for backwards compatibility):

- **Spatial Dead-Zone** (`camera.spatial_deadzone_*`): Ball can sit in center ~30% of FOV without triggering a pan. Camera only moves as ball approaches frame edge. Replaces "tracking shot" feel with "broadcast operator" feel.
- **Kalman Velocity Lookahead** (`camera.lookahead_*`): Projects target 3-5 frames ahead using Kalman velocity state on fast passes. Self-regulating — negligible on slow play.
- **Velocity-Adaptive Blending** (`center_of_play.velocity_blend_*`): Continuous ball/cluster blend weight based on ball velocity. Fast ball = 95% ball weight, slow ball = 50/50 with cluster.

### Phase 2: TrackNetV3 Temporal Ball Detection

Optional dual-path detection mode:

- **TrackNetV3** (`detection.ball_model.type: tracknet`): Temporal heatmap model uses 3 consecutive frames to detect motion-blurred sub-10px balls that single-frame YOLO misses.
- YOLO continues handling player detection (class 0); TrackNetV3 handles ball detection when enabled.
- Ring buffer of 3 frames fed per-frame to TrackNetV3; output converted to synthetic bbox compatible with downstream pipeline.
- New module: `src/tracknet.py` (vendored encoder-decoder architecture).

### Phase 3: TrackNetV3 Training Pipeline

- `src/tracknet_data.py`: Converts YOLO-format labels to Gaussian heatmaps, provides PyTorch Dataset for frame triplets + heatmaps.
- `scripts/train_tracknet.py`: Training script with weighted focal loss, ReduceLROnPlateau scheduler, checkpoint saving.
- Dashboard endpoints for building TrackNetV3 datasets and launching training.

### Phase 4: Horizontal Strip Tiling

- Configurable NxM tile grid (`detector.tiling.grid: [rows, cols]`) replaces hardcoded 2x2.
- `[1, 4]` = 4 horizontal strips (ideal for 180+ panoramic).
- Equirectangular-aware overlap boost at horizontal edges (`detector.tiling.equirect_aware_overlap`, `edge_overlap_boost`).
- Bug fix: class ID now preserved from YOLO detection (was hardcoded to 0).

## External Services

- **Label Studio** (port 8080) — annotation interface
- **NVIDIA GPU** — Tesla P40 via nvidia-docker
- **FFmpeg** — streaming video I/O
