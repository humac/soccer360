# Soccer360 - Functional Discovery Notes

> Internal working document. Not a deliverable.

## Application Type

Soccer360 is a **CLI-based video processing pipeline** (not a web application). It processes equirectangular 360-degree soccer match recordings into broadcast-quality outputs. Deployed as a Docker container with a persistent file-watching daemon.

## User Roles

| Role | Description | Interface |
|------|-------------|-----------|
| **Operator** | Day-to-day user: ingests videos, monitors processing, retrieves outputs, labels hard frames | CLI commands, file system (ingest folder), Label Studio web UI |
| **Administrator** | Sets up server, configures pipeline, manages Docker, handles GPU/model lifecycle, troubleshoots | Shell access, Docker Compose, YAML config, scripts |

## Core Features by Role

### Operator Features

| Feature | How It Works | Config Section |
|---------|-------------|----------------|
| Video ingest | Drop `.mp4`/`.mov` into `/tank/ingest/` | `watcher`, `paths.ingest` |
| Automatic processing | Watcher daemon detects new files, runs pipeline | `watcher` |
| Manual processing | `soccer360 process <path>` for one-off runs | CLI |
| Output retrieval | Browse `/tank/processed/<match>/` for results | `paths.processed` |
| Highlight clips | Auto-generated in `/tank/highlights/<match>/` | `highlights` |
| Hard frame labeling | Label Studio at `http://<server>:8080` | `active_learning` |
| Processing logs | `docker compose logs -f worker` | `logging` |

### Administrator Features

| Feature | How It Works | Config Section |
|---------|-------------|----------------|
| Installation | `scripts/install.sh` | N/A |
| Container verification | `make verify-container-assets` | N/A |
| Model management | Place models in `/tank/models/`, configure resolution | `model`, `detector`, `detection` |
| Training | `scripts/train_ball.sh [epochs]` | `configs/model_config.yaml` |
| Dataset building | `scripts/build_dataset.sh` | N/A |
| GPU configuration | Docker runtime, CUDA device selection | `detection.device`, compose env |
| Archival policy | Move/copy/leave after processing | `ingest` |
| Dedupe management | Reset state file to force reprocessing | `watcher.processed_state_file` |
| Health monitoring | Docker healthchecks, log inspection | `logging` |

## Data Model (File-Based)

No traditional database. All state is file-based:

### Input
- `/tank/ingest/*.mp4` — raw 360-degree video files

### Processing Artifacts (per match)
- `detections.jsonl` — per-frame YOLO ball detections
- `tracks.json` — tracked/stabilized ball positions
- `camera_path.json` — per-frame virtual camera (yaw, pitch, FOV)
- `foi_meta.json` — Field-of-Interest metadata
- `hard_frames.json` — hard-frame candidates with trigger reasons
- `metadata.json` — pipeline execution record (mode, timings, status)

### Output (per match)
- `broadcast.mp4` — auto-follow broadcast-style view
- `tactical_wide.mp4` — fixed wide-angle tactical view
- `highlight_*.mp4` — highlight clips (normal mode only)

### Active Learning
- `/tank/labeling/<match>/frames/` — exported hard-frame JPEGs
- `/tank/labeling/<match>/labels/` — YOLO-format annotations
- `/tank/labeling/<match>/hard_frames.json` — manifest
- `/tank/labeling/dataset/` — consolidated train/val dataset

### Persistent State
- `watcher_processed_ingest.json` — dedupe fingerprints (prevents reprocessing)
- `/tank/models/ball_best.pt` — active fine-tuned model
- `/tank/archive_raw/` — archived original recordings

## Runtime Modes

1. **V1 Bootstrap** (`detection` section in config) — full pipeline with YOLO + BallStabilizer + active learning
2. **Legacy** (no `detection` section) — full pipeline with ByteTrack tracker
3. **NO_DETECT** (model unavailable + `allow_no_model: true`) — static camera, broadcast + tactical only

## External Integrations

- **Label Studio** — separate Docker service for annotation (port 8080)
- **NVIDIA GPU** — Tesla P40 via nvidia-docker runtime
- **FFmpeg** — streaming video I/O (system binary, not Python package)

## Configuration Surface

Single config file: `configs/pipeline.yaml` with 17 top-level sections covering paths, model, detection, tracking, camera, rendering, highlights, archival, and active learning.

## Key Operational Paths

```
/tank/ingest/          -> queue folder (input)
/scratch/work/         -> NVMe temp space (auto-cleaned)
/tank/processed/       -> final outputs
/tank/highlights/      -> highlight clips
/tank/models/          -> YOLO weights
/tank/labeling/        -> hard frames + labels
/tank/archive_raw/     -> archived originals
/tank/logs/            -> pipeline logs
```
