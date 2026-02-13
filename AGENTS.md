# Soccer360 - AI Agent Context

Reference context for coding agents working in this repository.

## Project Overview

Soccer360 processes equirectangular 360 soccer video into:

- `broadcast.mp4` (auto-follow view)
- `tactical_wide.mp4` (fixed tactical view)
- `highlight_*.mp4` clips (normal mode only)
- run artifacts (`detections.jsonl`, `tracks.json`, `camera_path.json`, `foi_meta.json`, `hard_frames.json`, `metadata.json`)

## Current Pipeline Modes

The orchestrator is in `src/pipeline.py` and supports three runtime modes:

1. `v1 bootstrap mode` (default config includes `detection` section):

- Phase 1: `Detector.run_streaming()` (YOLO, FoI, y-band filter, best-per-frame)
- Phase 2: `BallStabilizer.run()` (temporal persistence/jump-speed rejection)
- Phase 2.5: `ActiveLearningExporter.run()` (low-conf, lost-run, jump-reject hard frames)
- Phase 3+: camera, broadcast render, tactical render, highlights, export, cleanup

1. `legacy mode` (if `detection` section is removed):

- Phase 2 uses `Tracker.run()` (ByteTrack)
- Phase 2.5 uses `HardFrameExporter.run()`

1. `no_detect mode`:

- Triggered when model resolution fails and `mode.allow_no_model: true`
- Skips detection/tracking/highlights, produces static-camera broadcast + tactical outputs

## Source Layout

```text
src/
  pipeline.py        Orchestrator; mode selection and phase coordination
  detector.py        YOLO streaming inference, FoI filter, model resolution
  tracker.py         Legacy ByteTrack tracker + V1 BallStabilizer
  active_learning.py V1 hard-frame candidate selection and export
  hard_frames.py     Legacy hard-frame export
  camera.py          Camera path smoothing (Kalman + EMA + deadband + dynamic FOV)
  reframer.py        360->perspective rendering (parallel segments with overlap)
  highlights.py      Heuristic highlight detection and clip export
  exporter.py        Final outputs, metadata, ingest archival bookkeeping
  watcher.py         Ingest daemon + persistent dedupe state
  trainer.py         Fine-tuning + TensorRT export + manual hard-frame export command
  cli.py             Click CLI entrypoint
  utils.py           ffmpeg streaming I/O, angle math, JSON/JSONL helpers
```

## Model Resolution

`src/detector.py` has separate resolvers:

- V1 (`resolve_model_path_v1`):

1. `/app/models/ball_best.pt`
2. `/app/yolov8s.pt`
3. `None` only when `mode.allow_no_model: true` (NO_DETECT)
4. otherwise raise `RuntimeError`

- Legacy (`resolve_model_path`):

1. `{paths.models}/ball_best.pt`
2. `model.path`
3. `/app/models/ball_base.pt` (copied into tank models)
4. `None` (NO_DETECT)

## Critical Math and Conventions

- Equirectangular conversion (`src/utils.py`):

```python
yaw = (x / width) * 360.0 - 180.0
pitch = 90.0 - (y / height) * 180.0
```

- Angle wrapping is always `(-180, 180]`:

```python
a = a % 360.0
if a > 180.0:
    a -= 360.0
```

- Vertical FOV (`src/reframer.py`) must be tangent-based:

```python
fov_v = degrees(2 * atan(tan(radians(fov_h / 2)) * (out_h / out_w)))
```

- FoI auto-center (`src/detector.py`): 72-bin (5 deg) yaw histogram over early detections, peak-bin cluster, circular mean.

## Configuration

Primary runtime config: `configs/pipeline.yaml`.

Key sections currently used in production:

- `paths`, `model`, `detector`, `field_of_interest`, `tracker`, `camera`, `reframer`, `highlights`, `exporter`, `watcher`, `ingest`, `active_learning`, `detection`, `filters`, `tracking`, `mode`, `logging`

When adding/changing config:

1. Update `configs/pipeline.yaml`
2. Wire defaults in the owning module `__init__`
3. Update `tests/conftest.py` fixture config

## Ingest and Watcher Behavior

`src/watcher.py` maintains persistent dedupe state keyed by source-path + material fingerprint.

- State file configured by `watcher.processed_state_file` (relative paths resolve under `<paths.processed>/.state/`)
- `watcher.processed_state_max_entries` bounds state growth (0 means unlimited)
- Dedupe bookkeeping failures are logged but must not block processing

`src/exporter.py` performs post-success ingest archival per `ingest.*` config (`move`/`copy`/`leave`, collision policy `suffix`/`skip`/`overwrite`) and records outcome in `metadata.json`.

## Build / Install Verification (Recent Fixes)

Canonical worker build verification is `scripts/verify_container_assets.sh`:

- Docker preflight: checks `docker` CLI exists and daemon is reachable
- Deps sync guard: validates `requirements-docker.txt` vs `pyproject.toml`
- Host dependency-check fallback: if host `python3` missing or missing `tomllib`/`tomli`, fallback runs in `python:3.11-slim`
- Failure behavior: mismatch details are printed; docker fallback execution failures are distinguished from true dependency mismatches
- Build mode flags:
  - `NO_CACHE=1` -> `--no-cache`
  - `RESET=1` -> always `docker compose down --remove-orphans` before build
- BuildKit forced for compose builds
- Asserts rebuilt image SHA equals ephemeral container SHA and validates `/app/yolov8s.pt` and `/app/.ultralytics` runtime permissions

`scripts/install.sh` now uses this verifier as the canonical worker-image build path and uses the configured compose project name.

## Testing

Run tests in Docker:

```bash
docker compose run --rm worker pytest tests/ -v
```

Useful maintenance checks:

```bash
make verify-container-assets
NO_CACHE=1 RESET=1 bash scripts/verify_container_assets.sh
make check-deps-sync
```

## Common Pitfalls

1. No local venv workflow: run test/runtime commands in Docker.
2. Do not break the `(-180, 180]` angle convention.
3. Tracker/stabilizer thresholds are in detector-space pixels, not source resolution.
4. Keep frame I/O streaming (ffmpeg pipes); do not introduce disk frame dumps.
5. Reframer segment workers are module-level callables for process pickling.
6. Preserve overlap semantics at segment boundaries (`reframer.overlap_sec`).
