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

33 endpoints total (unauthenticated): status, jobs, decisions, training/labeling, staging/upload, media/video streaming, SSE event stream.

## Configuration

Single file: `configs/pipeline.yaml` with 15+ sections: paths, detector, detection, camera, center_of_play, tracker, highlights, active_learning, reframer, exporter, ingest, watcher, dashboard, logging.

## External Services
- **Label Studio** (port 8080) — annotation interface
- **NVIDIA GPU** — Tesla P40 via nvidia-docker
- **FFmpeg** — streaming video I/O
