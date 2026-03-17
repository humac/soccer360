# Soccer360 - Claude Code Context

Short, implementation-accurate context for Claude-style agents.
`AGENTS.md` is the canonical long-form reference.

## Current Implementation Snapshot

Soccer360 is a two-pass 360 video pipeline producing `broadcast.mp4`, `tactical_wide.mp4`, highlights, and run artifacts.

Runtime modes in `src/pipeline.py`:

- **V1 bootstrap mode** (`detection` section present): `Detector` -> `BallStabilizer` -> `ActiveLearningExporter` -> `PlayerClusterComputer` (if enabled) -> camera/reframe/highlights/export.
- **Legacy mode** (`detection` section absent): `Detector` -> `Tracker` (ByteTrack) -> `HardFrameExporter` -> `PlayerClusterComputer` (if enabled) -> camera/reframe/highlights/export.
- **NO_DETECT mode**: static camera path + broadcast/tactical only (no detect/track/highlights).

## Key Files

- `src/pipeline.py`: mode resolution, phase orchestration, event bus integration, decision hooks
- `src/detector.py`: model resolution, FoI, V1/legacy detection behavior
- `src/tracker.py`: ByteTrack (legacy) + BallStabilizer (V1) — filters to class 32 (ball) only
- `src/player_cluster.py`: center-of-play estimation from player cluster positions (class 0)
- `src/highlights.py`: heuristic highlight detection — ball-based (speed, goal-box, direction) + cluster-based (convergence, velocity, goal zone, density) + scoring/ranking
- `src/active_learning.py`: V1 hard-frame export triggers/gating
- `src/watcher.py`: ingest queue daemon + persistent dedupe fingerprints + EventBus creation
- `src/exporter.py`: metadata + ingest archival (`move`/`copy`/`leave`, collision policy)
- `src/events.py`: EventStore (SQLite) + EventBus (null-safe pipeline wrapper) + decision queue + stale job cleanup on startup
- `src/dashboard.py`: FastAPI monitoring dashboard + REST API + SSE stream + training management
- `src/camera.py`: camera path generation — hybrid ball+cluster → Kalman → EMA → deadband → FOV EMA smoothing
- `src/metrics.py`: PhaseTimer (context-manager timing) + gpu_utilization_snapshot (nvidia-smi) + cpu_ram_snapshot (/proc)
- `src/static/index.html`: single-page dashboard UI (vanilla JS/CSS, EventSource SSE)
- `scripts/verify_container_assets.sh`: canonical container build/runtime verifier
- `scripts/install.sh`: calls verifier as canonical worker build path

## Recent Operational Fixes Reflected In Repo

- Verifier preflights Docker CLI/daemon before fallback operations.
- Deps-sync check (`requirements-docker.txt` vs `pyproject.toml`) now:
  - captures deterministic host exit code
  - falls back to Docker when host `python3` is missing or missing `tomllib`/`tomli`
  - prints mismatch output before failing
  - distinguishes true mismatch from fallback execution failure
- `RESET=1` now triggers `docker compose down --remove-orphans` regardless of `NO_CACHE`.
- BuildKit is forced in verifier builds.
- `install.sh` routes worker-image build through verifier and honors compose project naming.
- Worker remains numeric `1000:1000`; image provides UID/GID 1000 passwd/group compatibility plus `HOME`/`USER`/`LOGNAME` to avoid torch/getpass crashes.
- Verifier now asserts `python -c "import getpass; print(getpass.getuser())"` succeeds at runtime.
- V1 model-path precedence is explicit and logged once per job:
  - `detector.model_path` > `detection.path` > `default`
  - explicit non-default `detector.model_path` must point to an existing file (else resolver raises `RuntimeError`)
  - source enum: `detector.model_path`, `detection.path`, `default`
  - runtime log format: `Model resolved: <path> (source=<source>)`
- Dockerfile pins Pascal-safe PyTorch from cu121 (`torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `torchaudio==2.4.1+cu121`) and constrains requirements install to that trio.
- Verifier now prints torch/CUDA + GPU capability diagnostics, treats arch-list mismatch as warning, and uses CUDA conv2d smoke as the authoritative gate (`GPU_SMOKE=1` default, `GPU_SMOKE=0` to skip).
- Verifier resolves model path in-container using runtime Python logic (`src.utils.load_config` + `resolve_v1_model_path_and_source`), emits only `CONFIG_PATH`/`MODEL_PATH`/`MODEL_SOURCE` on stdout, validates selected `MODEL_PATH` via `test -s`, and only enforces baked `/app/yolov8s.pt` checks when that path is actually selected.
- Resolver failures are fail-fast and include attempted `CONFIG_PATH`, resolver exit code, and captured stderr. Use `VERBOSE=1` to print captured resolver stderr/noise diagnostics when non-empty.
- Resolver exit codes are deterministic: `11` (config path/readability), `12` (config parse/load), `13` (resolver import/runtime resolution).
- Canonical explicit Roboflow path is `/app/models/roboflow/football_players_v1.pt`; in default compose runtime `/app/models` is mounted from host `/tank/models`, so place weights at `/tank/models/roboflow/football_players_v1.pt`.

## Center of Play (Hybrid Camera Tracking)

- Detects players (COCO class 0) alongside ball (class 32) in same YOLO pass
- `src/player_cluster.py`: `PlayerClusterComputer` computes per-frame trimmed-mean centroid from player positions
- Pipeline Phase 2.7: runs after tracking, before camera path generation
- Camera blending priority: ball (high conf, 85%/15%) > ball+cluster (low conf, 50%/50%) > cluster only > drift-to-center
- FOV adapts to player spread: wide spread → wider FOV, tight cluster → tighter FOV
- Config section: `center_of_play:` with `enabled`, `player_class`, `min_player_conf`, `trim_fraction`, `min_players`, `ball_blend_weight`, `ema_alpha`, `fov_from_spread`, `spread_max_fov`, `spread_min_deg`, `spread_max_deg`
- Detection model: YOLO26l (`yolo26l.pt`), COCO-pretrained, `classes: [32, 0]`, `max_det: 50`
- Tracker/BallStabilizer filter to class 32 only — person detections do not affect ball tracking
- Output: `player_cluster.json` (per-frame centroid, spread, player count)

## Camera Smoothing Pipeline

Multi-stage smoothing in `src/camera.py` prevents jitter in the broadcast output:

1. Hybrid ball+cluster blending (priority: ball high-conf > ball+cluster > cluster only > drift)
2. Kalman filter (4-state: yaw, pitch, velocity) with configurable process/measurement noise
3. EMA post-smoothing on yaw/pitch (`camera.ema_alpha`)
4. Deadband + smooth velocity gain ramp (linear interpolation, not binary threshold)
5. Smooth pan speed interpolation between normal and fast limits (no binary flicker)
6. **FOV EMA smoothing** (`camera.fov_ema_alpha: 0.08`) — prevents zoom oscillation; lost→found FOV transitions are gradual, not instant
7. Spread data carryforward across cluster gaps (no FOV drops on missing frames)

Key config: `camera.fov_ema_alpha`, `camera.deadband_deg`, `camera.velocity_threshold_deg_per_sec`, `camera.ema_alpha`

## Monitoring Dashboard

- **Port 8088** (avoids Label Studio on 8080)
- `soccer360 dashboard` CLI command starts FastAPI + uvicorn
- `dashboard` Docker Compose service reuses `soccer360-worker:local` image
- EventBus usage in pipeline is always guarded by `if self.event_bus:` — CLI path unchanged
- SQLite WAL mode store at `dashboard.db_path` (default `/tank/data/dashboard.db`)
- SSE endpoint (`/api/events`) streams phase events, GPU snapshots, system snapshots (CPU/RAM), decisions, status heartbeats
- `/api/system` endpoint + `system_snapshot` SSE events provide CPU utilization, RAM usage, core count (from `/proc`)
- Dashboard UI: GPU gauges, System card (CPU/RAM gauges), pipeline progress, job history, training management
- Decision hooks in pipeline: mode confirmation (30s), post-detection review (60s), hard frame labeling (120s)
- Training management API: `/api/training/labeling-status`, `/api/training/upload-labels/{match_name}`, `/api/training/build-dataset`, `/api/training/train`, `/api/training/models`
- Config section: `dashboard:` with `enabled`, `db_path`, `port`
- Dependencies: `fastapi>=0.109`, `uvicorn>=0.27`, `sse-starlette>=2.0`, `python-multipart>=0.0.6`

## Non-Negotiable Conventions

- Angle wrap convention: `(-180, 180]`
- Equirect mapping: `yaw = (x/W)*360-180`, `pitch = 90-(y/H)*180`
- Vertical FOV uses tangent formula, never linear approximation
- Streaming architecture: ffmpeg pipes only, no intermediate frame dumps
- Pixel thresholds are detector-space pixels

## Config + Test Workflow

- Runtime config: `configs/pipeline.yaml`
- If adding config keys: update config file, module defaults, and `tests/conftest.py`
- Test in Docker:

```bash
docker compose run --rm worker pytest tests/ -v
```

Compose service entrypoint is `soccer360`; for Python checks use:

```bash
docker compose run --rm --no-deps --entrypoint python worker -c "import torch; print(torch.__version__)"
```

To print resolved model path/source (same logic as verifier):

```bash
docker compose run --rm --no-deps --entrypoint python worker -c "import os; from src.utils import load_config; from src.detector import resolve_v1_model_path_and_source; config_path=(os.getenv('SOCCER360_CONFIG') or '/app/configs/pipeline.yaml'); cfg=load_config(config_path); p,s=resolve_v1_model_path_and_source(cfg, models_dir=cfg.get('paths', {}).get('models', '/app/models')); print(f'CONFIG_PATH={config_path}'); print(f'MODEL_PATH={p}'); print(f'MODEL_SOURCE={s}')"
```
