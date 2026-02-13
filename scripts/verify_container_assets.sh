#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT="${PROJECT:-soccer360}"
SERVICE="${SERVICE:-worker}"
IMAGE_TAG="${IMAGE_TAG:-soccer360-worker:local}"
NO_CACHE="${NO_CACHE:-0}"
RESET="${RESET:-0}"
SKIP_DEPS_SYNC="${SKIP_DEPS_SYNC:-0}"
GPU_SMOKE="${GPU_SMOKE:-1}"
VERBOSE="${VERBOSE:-0}"
cid=""

log() {
  echo "[verify-container-assets] $*"
}

fail() {
  echo "[verify-container-assets] ERROR: $*" >&2
  exit 1
}

run_deps_sync_in_docker() {
  local docker_rc=0
  local docker_output

  docker_output="$(docker run --rm -v "$REPO_DIR":/repo -w /repo python:3.11-slim \
    python scripts/check_deps_sync.py 2>&1)" || docker_rc=$?
  if [ "$docker_rc" -eq 0 ]; then
    return 0
  fi
  if [ -n "$docker_output" ]; then
    echo "$docker_output" >&2
  fi
  if [ "$docker_rc" -eq 1 ]; then
    fail "requirements-docker.txt is out of sync with pyproject.toml"
  fi
  fail "deps sync docker fallback failed (exit $docker_rc)"
}

cleanup() {
  if [ -n "${cid:-}" ]; then
    docker rm -f "$cid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

cd "$REPO_DIR"

# --- Docker preflight ---
if ! command -v docker >/dev/null 2>&1; then
  fail "docker CLI not found in PATH"
fi
if ! docker info >/dev/null 2>&1; then
  fail "docker daemon unavailable or insufficient permissions"
fi

# --- Mode selection ---
build_args=()
if [ "$NO_CACHE" = "1" ]; then
  log "Mode: CLEAN (no-cache rebuild)"
  build_args+=(--no-cache)
else
  log "Mode: FAST (cached build, no service disruption)"
fi
if [ "$RESET" = "1" ]; then
  log "Reset compose state before build (project=$PROJECT)"
  docker compose -p "$PROJECT" down --remove-orphans
fi

# --- Deps sync guard ---
if [ "$SKIP_DEPS_SYNC" = "1" ]; then
  log "Deps sync check: SKIPPED (SKIP_DEPS_SYNC=1)"
else
  log "Deps sync check: verifying requirements-docker.txt matches pyproject.toml"
  sync_rc=0
  sync_output="$(python3 "$REPO_DIR/scripts/check_deps_sync.py" 2>&1)" || sync_rc=$?
  if [ "$sync_rc" -eq 0 ]; then
    log "Deps sync check: OK (host python3)"
  else
    if [ "$sync_rc" -eq 2 ]; then
      log "Deps sync check: host python3 missing tomllib/tomli, falling back to docker"
      run_deps_sync_in_docker
      log "Deps sync check: OK (docker fallback)"
    elif [ "$sync_rc" -eq 127 ]; then
      log "Deps sync check: host python3 not available, falling back to docker"
      run_deps_sync_in_docker
      log "Deps sync check: OK (docker fallback)"
    else
      if [ -n "$sync_output" ]; then
        echo "$sync_output" >&2
      fi
      fail "requirements-docker.txt is out of sync with pyproject.toml"
    fi
  fi
fi

# --- Build ---
before_id="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}' 2>/dev/null || true)"
log "before_id=${before_id:-<none>}"

log "BuildKit: forced on (DOCKER_BUILDKIT=1, COMPOSE_DOCKER_CLI_BUILD=1)"
log "Building worker image (project=$PROJECT)"
if ! DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 \
     docker compose -p "$PROJECT" build "${build_args[@]}" --pull=false "$SERVICE"; then
  log "Retrying build without --pull=false (compose compatibility fallback)"
  DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 \
    docker compose -p "$PROJECT" build "${build_args[@]}" "$SERVICE" \
    || fail "build failed for service '$SERVICE'"
fi

# --- SHA assertion ---
after_id="$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}' 2>/dev/null || true)"
if [ -z "$after_id" ] || [ "$after_id" = "<no value>" ]; then
  fail "could not resolve rebuilt image SHA for tag '$IMAGE_TAG'"
fi
log "project=$PROJECT"
log "image_tag=$IMAGE_TAG"
log "rebuilt_sha=$after_id"

log "Compose image mapping"
if ! docker compose -p "$PROJECT" config --images; then
  docker compose -p "$PROJECT" images
fi

# --- Ephemeral container ---
log "Start short-lived worker container"
cid="$(docker compose -p "$PROJECT" run -d --no-TTY --no-deps --entrypoint sleep "$SERVICE" 60)" \
  || fail "failed to start verifier container"

container_ref="$(docker inspect -f '{{.Config.Image}}' "$cid")" \
  || fail "failed to inspect container image ref"
container_sha="$(docker inspect -f '{{.Image}}' "$cid")" \
  || fail "failed to inspect container image SHA"
log "container_image_ref=$container_ref"
log "container_image_sha=$container_sha"
if [ "$container_sha" != "$after_id" ]; then
  fail "container image SHA does not match rebuilt image SHA"
fi

# --- Runtime checks ---
log "Resolve configured model path via in-container runtime logic"
resolver_rc=0
resolver_stdout_file="$(mktemp)"
resolver_stderr_file="$(mktemp)"
if docker exec -i \
  -e SOCCER360_CONFIG="${SOCCER360_CONFIG:-}" \
  -e VERBOSE="$VERBOSE" \
  "$cid" python - >"$resolver_stdout_file" 2>"$resolver_stderr_file" <<'PY'
import io
import os
import sys
import warnings
from pathlib import Path


def _stderr(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def _showwarning(message, category, filename, lineno, file=None, line=None):
    rendered = warnings.formatwarning(message, category, filename, lineno, line).rstrip("\n")
    _stderr(rendered)


out = sys.stdout
captured_stdout = io.StringIO()
sys.stdout = captured_stdout
warnings.showwarning = _showwarning
warnings.simplefilter("default")

verbose = (os.getenv("VERBOSE", "0").strip() == "1")
config_path = (os.getenv("SOCCER360_CONFIG") or "").strip() or "/app/configs/pipeline.yaml"
model_path = ""
model_source = ""
exit_code = 0

try:
    config_file = Path(config_path)
    if not config_file.exists():
        _stderr(f"resolver error: config file not found: {config_path}")
        exit_code = 11
    elif not config_file.is_file():
        _stderr(f"resolver error: config path is not a file: {config_path}")
        exit_code = 11
    elif not os.access(config_file, os.R_OK):
        _stderr(f"resolver error: config file is not readable: {config_path}")
        exit_code = 11

    if exit_code == 0:
        try:
            from src.utils import load_config
        except Exception as exc:  # pragma: no cover - runtime guard
            _stderr(f"resolver error: failed to import src.utils.load_config: {exc}")
            exit_code = 13

    if exit_code == 0:
        try:
            from src.detector import resolve_v1_model_path_and_source
        except Exception as exc:  # pragma: no cover - runtime guard
            _stderr(
                "resolver error: failed to import "
                f"src.detector.resolve_v1_model_path_and_source: {exc}"
            )
            exit_code = 13

    if exit_code == 0:
        try:
            cfg = load_config(config_path)
        except Exception as exc:
            _stderr(f"resolver error: failed to load/parse config: {exc}")
            exit_code = 12

    if exit_code == 0:
        try:
            models_dir = cfg.get("paths", {}).get("models", "/app/models")
            resolved_path, resolved_source = resolve_v1_model_path_and_source(
                cfg, models_dir=models_dir
            )
            model_path = "" if resolved_path is None else str(resolved_path)
            model_source = "" if resolved_source is None else str(resolved_source)
        except Exception as exc:
            _stderr(f"resolver error: failed to resolve model path/source: {exc}")
            exit_code = 13
finally:
    noise = captured_stdout.getvalue()
    if noise and (exit_code != 0 or verbose):
        _stderr("resolver stdout-noise begin")
        for noise_line in noise.splitlines():
            _stderr(noise_line)
        _stderr("resolver stdout-noise end")

    out.write(f"CONFIG_PATH={config_path}\n")
    out.write(f"MODEL_PATH={model_path if exit_code == 0 else ''}\n")
    out.write(f"MODEL_SOURCE={model_source if exit_code == 0 else ''}\n")
    out.flush()
    raise SystemExit(exit_code)
PY
then
  resolver_rc=0
else
  resolver_rc=$?
fi
resolver_stdout="$(cat "$resolver_stdout_file")"
resolver_stderr="$(cat "$resolver_stderr_file")"
rm -f "$resolver_stdout_file" "$resolver_stderr_file"

resolved_config_path=""
resolved_model_path=""
resolved_model_source=""
seen_config=0
seen_model_path=0
seen_model_source=0
while IFS= read -r raw_line; do
  line="${raw_line%$'\r'}"
  case "$line" in
    CONFIG_PATH=*)
      value="${line#*=}"
      resolved_config_path="$value"
      seen_config=1
      ;;
    MODEL_PATH=*)
      if [ "$resolver_rc" -eq 0 ]; then
        value="${line#*=}"
        resolved_model_path="$value"
        seen_model_path=1
      fi
      ;;
    MODEL_SOURCE=*)
      if [ "$resolver_rc" -eq 0 ]; then
        value="${line#*=}"
        resolved_model_source="$value"
        seen_model_source=1
      fi
      ;;
  esac
done <<< "$resolver_stdout"

if [ -z "${resolved_config_path}" ]; then
  resolved_config_path="${SOCCER360_CONFIG:-/app/configs/pipeline.yaml}"
fi

if [ "$resolver_rc" -eq 0 ] && [ "$VERBOSE" = "1" ] && [ -n "${resolver_stderr//[[:space:]]/}" ]; then
  log "Resolver stderr (VERBOSE=1):"
  while IFS= read -r err_line; do
    [ -n "$err_line" ] && log "resolver_stderr=$err_line"
  done <<< "$resolver_stderr"
fi

if [ "$resolver_rc" -ne 0 ]; then
  if [ -n "${resolver_stderr//[[:space:]]/}" ]; then
    echo "[verify-container-assets] resolver stderr:" >&2
    echo "$resolver_stderr" >&2
  fi
  fail "model resolver failed (exit=$resolver_rc, config_path=$resolved_config_path). Re-run with VERBOSE=1 for additional diagnostics."
fi

log "resolved_config_path=$resolved_config_path"
log "resolved_model_path=${resolved_model_path:-<empty>}"
log "resolved_model_source=${resolved_model_source:-<empty>}"

if [ "$seen_config" -ne 1 ] || [ "$seen_model_path" -ne 1 ] || [ "$seen_model_source" -ne 1 ]; then
  if [ -n "${resolver_stderr//[[:space:]]/}" ]; then
    echo "[verify-container-assets] resolver stderr:" >&2
    echo "$resolver_stderr" >&2
  fi
  fail "resolver output contract violated (config_path=$resolved_config_path, exit=$resolver_rc). Expected CONFIG_PATH/MODEL_PATH/MODEL_SOURCE lines."
fi

if [ -z "${resolved_model_path//[[:space:]]/}" ] || [ -z "${resolved_model_source//[[:space:]]/}" ]; then
  if [ -n "${resolver_stderr//[[:space:]]/}" ]; then
    echo "[verify-container-assets] resolver stderr:" >&2
    echo "$resolver_stderr" >&2
  fi
  fail "model resolver returned empty MODEL_PATH/MODEL_SOURCE (config_path=$resolved_config_path, exit=$resolver_rc). Re-run with VERBOSE=1 to print captured resolver stderr."
fi

case "$resolved_model_source" in
  detector.model_path|detection.path|default)
    ;;
  *)
    fail "invalid MODEL_SOURCE='$resolved_model_source' (config_path=$resolved_config_path)"
    ;;
esac

docker exec "$cid" sh -lc 'test -s "$1"' sh "$resolved_model_path" \
  || fail "resolved model path missing/empty in container: $resolved_model_path (config_path=$resolved_config_path)"
resolved_model_size="$(docker exec "$cid" sh -lc 'stat -c "%s" "$1"' sh "$resolved_model_path")" \
  || fail "failed to stat resolved model path: $resolved_model_path"
log "resolved_model_size_bytes=$resolved_model_size"

log "Runtime asset checks"
docker exec "$cid" bash -lc '
set -euo pipefail
id
stat -c "%u:%g %A %n" /app/.ultralytics
test -w /app/.ultralytics
' || fail "runtime asset checks failed"

if [ "$resolved_model_path" = "/app/yolov8s.pt" ]; then
  log "Resolved model is baked default; enforcing /app/yolov8s.pt checks"
  docker exec "$cid" bash -lc '
set -euo pipefail
ls -lah /app/yolov8s.pt
stat -c "%u:%g %A %n" /app/yolov8s.pt
test -s /app/yolov8s.pt
' || fail "baked /app/yolov8s.pt checks failed"
else
  log "Resolved model is not /app/yolov8s.pt; skipping baked yolov8s.pt checks"
fi

log "Runtime user identity check"
runtime_user="$(docker exec "$cid" python -c "import getpass; print(getpass.getuser())")" \
  || fail "runtime getpass.getuser() check failed"
if [ -z "${runtime_user//[[:space:]]/}" ]; then
  fail "runtime getpass.getuser() returned empty username"
fi
log "runtime_user=$runtime_user"

log "Runtime GPU diagnostics (torch/cuda)"
if docker exec "$cid" bash -lc 'command -v nvidia-smi >/dev/null 2>&1'; then
  gpu_info="$(docker exec "$cid" nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true)"
  if [ -n "${gpu_info//[[:space:]]/}" ]; then
    log "nvidia_smi=$gpu_info"
  else
    log "nvidia_smi=available but no query output"
  fi
else
  log "nvidia_smi=not available in container"
fi

gpu_diag="$(
  docker exec -i "$cid" python - <<'PY'
import torch

print(f"TORCH_VERSION={torch.__version__}")
print(f"TORCH_CUDA={torch.version.cuda}")
avail = torch.cuda.is_available()
print(f"CUDA_AVAILABLE={1 if avail else 0}")
arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
print("ARCH_LIST=" + ",".join(arch_list))
if avail:
    major, minor = torch.cuda.get_device_capability()
    sm_tag = f"sm_{major}{minor}"
    print(f"DEVICE_CAP={major}.{minor}")
    print(f"SM_TAG={sm_tag}")
    print(f"ARCH_MATCH={1 if sm_tag in arch_list else 0}")
PY
)" || fail "torch cuda diagnostics failed"
while IFS= read -r line; do
  [ -n "$line" ] && log "$line"
done <<< "$gpu_diag"

cuda_available="$(printf '%s\n' "$gpu_diag" | awk -F= '$1=="CUDA_AVAILABLE" {print $2}')"
arch_match="$(printf '%s\n' "$gpu_diag" | awk -F= '$1=="ARCH_MATCH" {print $2}')"
sm_tag="$(printf '%s\n' "$gpu_diag" | awk -F= '$1=="SM_TAG" {print $2}')"

if [ "$cuda_available" = "1" ]; then
  if [ "${arch_match:-0}" != "1" ]; then
    log "WARNING: torch arch list does not include ${sm_tag:-<unknown>}."
    log "WARNING: this can be a false negative on some builds; CUDA smoke is the authoritative gate."
    log "WARNING: run GPU_SMOKE=1 make verify-container-assets for a definitive kernel test."
  fi

  if [ "$GPU_SMOKE" = "1" ]; then
    log "Runtime CUDA smoke test (GPU_SMOKE=1)"
    docker exec -i "$cid" python - <<'PY' || fail "runtime CUDA smoke test failed"
import torch
import torch.nn.functional as F

x = torch.randn(1, 3, 224, 224, device="cuda")
w = torch.randn(8, 3, 3, 3, device="cuda")
y = F.conv2d(x, w)
print("GPU_SMOKE conv2d ok", tuple(y.shape))
PY
  else
    log "Runtime CUDA smoke test skipped (GPU_SMOKE=0). Set GPU_SMOKE=1 for authoritative kernel validation."
  fi
else
  log "CUDA unavailable at runtime; skipping GPU arch and smoke checks."
fi

log "verify_container_assets: ok"
