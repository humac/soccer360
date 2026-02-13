# Soccer360 - Claude Code Context

Short, implementation-accurate context for Claude-style agents.
`AGENTS.md` is the canonical long-form reference.

## Current Implementation Snapshot

Soccer360 is a two-pass 360 video pipeline producing `broadcast.mp4`, `tactical_wide.mp4`, highlights, and run artifacts.

Runtime modes in `src/pipeline.py`:

- **V1 bootstrap mode** (`detection` section present): `Detector` -> `BallStabilizer` -> `ActiveLearningExporter` -> camera/reframe/highlights/export.
- **Legacy mode** (`detection` section absent): `Detector` -> `Tracker` (ByteTrack) -> `HardFrameExporter` -> camera/reframe/highlights/export.
- **NO_DETECT mode**: static camera path + broadcast/tactical only (no detect/track/highlights).

## Key Files

- `src/pipeline.py`: mode resolution and phase orchestration
- `src/detector.py`: model resolution, FoI, V1/legacy detection behavior
- `src/tracker.py`: ByteTrack (legacy) + BallStabilizer (V1)
- `src/active_learning.py`: V1 hard-frame export triggers/gating
- `src/watcher.py`: ingest queue daemon + persistent dedupe fingerprints
- `src/exporter.py`: metadata + ingest archival (`move`/`copy`/`leave`, collision policy)
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
