# Soccer360 - Gemini Context

Concise context for Gemini-style agents. `AGENTS.md` is the detailed source of truth.

## What Is Current

Soccer360 ingests 360 match video and outputs:

- `broadcast.mp4`
- `tactical_wide.mp4`
- `highlight_*.mp4` (normal mode)
- artifacts/metadata in processed output folders

## Runtime Modes in `src/pipeline.py`

1. **V1 bootstrap mode** (default config has `detection` section):

- detection + FoI + y-range filter + best-per-frame
- `BallStabilizer` temporal gating/rejection
- `ActiveLearningExporter` hard-frame export

1. **Legacy mode** (no `detection` section):

- ByteTrack-based `Tracker`
- legacy `HardFrameExporter`

1. **NO_DETECT mode**:

- no model available with `mode.allow_no_model: true`
- static camera path, skips detect/track/highlights

## Important Files

- `src/detector.py`: model resolution (`resolve_model_path_v1` and legacy resolver), FoI math
- `src/tracker.py`: ByteTrack and V1 stabilizer logic
- `src/active_learning.py`: V1 frame export triggers (`low_conf`, `lost_run`, `jump_reject`)
- `src/watcher.py`: ingest queue handling + persistent dedupe fingerprint store
- `src/exporter.py`: finalization + ingest archival status in `metadata.json`
- `scripts/verify_container_assets.sh`: canonical image/runtime verifier
- `scripts/install.sh`: uses verifier for worker build path

## Recent Fixes Reflected

- Verifier dependency sync is robust:
  - captures explicit host return code
  - Docker fallback when host `python3` or `tomllib`/`tomli` is unavailable
  - prints mismatch output before failing
  - distinguishes mismatch vs fallback execution failure
- Verifier preflights Docker availability (`command -v docker`, `docker info`)
- `RESET=1` compose reset is independent of `NO_CACHE`
- BuildKit is forced during verifier compose builds
- Install script uses verifier as canonical path

## Critical Conventions

- Angle convention is always `(-180, 180]`
- Equirect conversion: `yaw=(x/W)*360-180`, `pitch=90-(y/H)*180`
- Vertical FOV must use tangent formula
- Keep streaming ffmpeg design (no frame dumps to disk)
- Detector-space pixels drive tracking/stabilization thresholds

## Config + Testing

- Main config: `configs/pipeline.yaml`
- Keep config changes synced with module defaults and `tests/conftest.py`
- Tests run in Docker:

```bash
docker compose run --rm worker pytest tests/ -v
```
