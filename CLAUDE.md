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
