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
- `src/dashboard.py`: FastAPI dashboard — staging upload with progress/resume, processed-match list, single-match API, and match reset
- `src/static/match.html`: dedicated match playback page served at `/match/{name}` — two-column sidebar + video player, lifecycle stepper, lineage panel, reset preview; no polling
- `src/static/index.html`: SPA with four workspaces (Overview, Matches, Labeling & Training, Models & Files); all confirmations use in-page modals, all results use toast notifications — no native browser `alert()`/`confirm()` anywhere
- `src/static/detection_settings.html`: read-only config viewer
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
- Worker runtime remains numeric `1000:1000`; image now guarantees UID/GID 1000 passwd/group compatibility plus `HOME`/`USER`/`LOGNAME` for `getpass` safety.
- Verifier now checks `python -c "import getpass; print(getpass.getuser())"` inside the runtime container.
- V1 model-path precedence is explicit and logged once per job:
  - `detector.model_path` > `detection.path` > `default`
  - explicit non-default `detector.model_path` must point to an existing file (else resolver raises `RuntimeError`)
  - source enum: `detector.model_path`, `detection.path`, `default`
  - runtime log format: `Model resolved: <path> (source=<source>)`
- Dockerfile pins Pascal-compatible PyTorch from cu121 (`torch==2.4.1+cu121`, `torchvision==0.19.1+cu121`, `torchaudio==2.4.1+cu121`) and constrains requirements install to those versions.
- Verifier prints torch/CUDA + GPU capability diagnostics, treats arch-list mismatch as warning-only, and uses CUDA conv2d smoke as the authoritative gate (`GPU_SMOKE=1` by default, `GPU_SMOKE=0` to skip).
- Verifier resolves model path in-container using runtime Python logic (`src.utils.load_config` + `resolve_v1_model_path_and_source`), emits only `CONFIG_PATH`/`MODEL_PATH`/`MODEL_SOURCE` on stdout, validates selected `MODEL_PATH` with `test -s`, and only enforces baked `/app/models/yolo26l.pt` checks when that path is selected.
- Resolver failures are fail-fast and include attempted `CONFIG_PATH`, resolver exit code, and captured stderr. Use `VERBOSE=1` to print captured resolver stderr/noise diagnostics when non-empty.
- Resolver exit codes are deterministic: `11` (config path/readability), `12` (config parse/load), `13` (resolver import/runtime resolution).
- Canonical explicit Roboflow path is `/app/models/roboflow/football_players_v1.pt`; in default compose runtime `/app/models` is mounted from host `/tank/models`, so place weights at `/tank/models/roboflow/football_players_v1.pt`.

## Critical Conventions

- Angle convention is always `(-180, 180]`
- Equirect conversion: `yaw=(x/W)*360-180`, `pitch=90-(y/H)*180`
- Vertical FOV must use tangent formula
- Keep streaming ffmpeg design (no frame dumps to disk)
- Detector-space pixels drive tracking/stabilization thresholds

## Config + Testing

- Main config: `configs/pipeline.yaml`
- Keep config changes synced with module defaults and `tests/conftest.py`
- `paths.stagging` defaults to `/tank/stagging` for the UI-managed requeue flow
- `detection.img_size`: YOLO inference resolution — use `3840` for 8K source (quality-first; ~4-5hr total); `960`–`1920` for 4K/5.7K faster runs (~1-1.5hr)
- `reframer.output_resolution`: final output size — `[3840, 2160]` for 4K UHD, `[1920, 1080]` for 1080p
- `reframer.source_downscale`: set to `[3840, 1920]` to downscale before reframing for speed; `null` = full native resolution
- `exporter.encoder`: `nvenc` for NVIDIA hardware encoding, `cpu` for libx264/libx265
- Staging upload API: `POST /api/staging/upload` supports `X-Upload-Offset` header for resume (writes `.uploading` partial, atomically renames on completion); `GET /api/staging/upload-status?filename=`; `DELETE /api/staging/upload-cancel`
- Match playback: `GET /api/media/matches/{name}` returns single match metadata + video list; `GET /match/{name}` serves `match.html`
- Tests run in Docker:

```bash
docker compose run --rm --entrypoint python worker -m pytest tests/ -v
```

Fast host-side slice:

```bash
pytest tests/test_dashboard.py tests/test_events.py tests/test_watcher.py -q
```

Worker service entrypoint is `soccer360`; for Python diagnostics use:

```bash
docker compose run --rm --no-deps --entrypoint python worker -c "import torch; print(torch.__version__)"
```

To print resolved model path/source (same logic as verifier):

```bash
docker compose run --rm --no-deps --entrypoint python worker -c "import os; from src.utils import load_config; from src.detector import resolve_v1_model_path_and_source; config_path=(os.getenv('SOCCER360_CONFIG') or '/app/configs/pipeline.yaml'); cfg=load_config(config_path); p,s=resolve_v1_model_path_and_source(cfg, models_dir=cfg.get('paths', {}).get('models', '/app/models')); print(f'CONFIG_PATH={config_path}'); print(f'MODEL_PATH={p}'); print(f'MODEL_SOURCE={s}')"
```
