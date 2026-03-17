# Soccer360 - Functional Discovery Notes

> Internal working document. Not a deliverable.

## Application Type

Soccer360 is a **CLI-based video processing pipeline** (not a web application). It processes equirectangular 360-degree soccer match recordings into broadcast-quality outputs. Deployed as a Docker container with a persistent file-watching daemon.

## User Roles

| Role | Description | Interface |
| ---- | ----------- | --------- |
| **Operator** | Day-to-day user: ingests videos, monitors processing, retrieves outputs, labels hard frames | CLI commands, file system (ingest folder), Label Studio web UI |
| **Administrator** | Sets up server, configures pipeline, manages Docker, handles GPU/model lifecycle, troubleshoots | Shell access, Docker Compose, YAML config, scripts |

## Core Features by Role

### Operator Features

| Feature | How It Works | Config Section |
| ------- | ----------- | -------------- |
| Video ingest | Drop `.mp4`/`.mov` into `/tank/ingest/` | `watcher`, `paths.ingest` |
| Automatic processing | Watcher daemon detects new files, runs pipeline | `watcher` |
| Manual processing | `soccer360 process <path>` for one-off runs | CLI |
| Output retrieval | Browse `/tank/processed/<match>/` for results | `paths.processed` |
| Highlight clips | Auto-generated in `/tank/highlights/<match>/` | `highlights` |
| Hard frame labeling | Label Studio at `http://<server>:8080` | `active_learning` |
| Processing logs | `docker compose logs -f worker` | `logging` |

### Administrator Features

| Feature | How It Works | Config Section |
| ------- | ----------- | -------------- |
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

- `detections.jsonl` — per-frame YOLO detections (ball class 32 + person class 0)
- `tracks.json` — tracked/stabilized ball positions (ball only)
- `player_cluster.json` — per-frame player cluster centroid and spread (center-of-play)
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

1. **V1 Bootstrap** (`detection` section in config) — full pipeline with YOLO + BallStabilizer + active learning + center-of-play
2. **Legacy** (no `detection` section) — full pipeline with ByteTrack tracker + center-of-play
3. **NO_DETECT** (model unavailable + `allow_no_model: true`) — static camera, broadcast + tactical only

## Center of Play (Hybrid Camera Tracking)

Detects players (COCO class 0) alongside the ball (class 32) in the same YOLO pass. A `PlayerClusterComputer` module computes a per-frame trimmed-mean centroid of player positions. The camera path generator blends ball tracking with player cluster positions:

- Ball detected (high conf): 85% ball + 15% cluster
- Ball detected (low conf): 50% ball + 50% cluster
- Ball lost, cluster available: 100% cluster (follows play instead of drifting to field center)
- Ball lost, no cluster: drift to field center (existing behavior)

FOV adapts to player spread (wider when players are spread across the pitch).

Config section: `center_of_play:` with `enabled`, `player_class`, `min_player_conf`, `trim_fraction`, `min_players`, `ball_blend_weight`, `ema_alpha`, `fov_from_spread`, `spread_max_fov`, `spread_min_deg`, `spread_max_deg`.

Training remains ball-only — person detection rides on the pretrained COCO model and is not retrained.

## Configuration Surface

Single config file: `configs/pipeline.yaml` with 18 top-level sections covering paths, model, detection, tracking, camera, center-of-play, rendering, highlights, archival, and active learning.

## External Integrations

- **Monitoring Dashboard** — FastAPI-based web UI (port 8088) for real-time pipeline monitoring, GPU/CPU/RAM gauges, decision handling, job history, active learning management (import, upload labels, build dataset, train)
- **Label Studio** — separate Docker service for annotation (port 8080)
- **NVIDIA GPU** — Tesla P40 via nvidia-docker runtime
- **FFmpeg** — streaming video I/O (system binary, not Python package)

## Key Operational Paths

```text
/tank/ingest/          -> queue folder (input)
/scratch/work/         -> NVMe temp space (auto-cleaned)
/tank/processed/       -> final outputs
/tank/highlights/      -> highlight clips
/tank/models/          -> YOLO weights
/tank/labeling/        -> hard frames + labels
/tank/archive_raw/     -> archived originals
/tank/logs/            -> pipeline logs
```
