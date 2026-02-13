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
cid=""

log() {
  echo "[verify-container-assets] $*"
}

fail() {
  echo "[verify-container-assets] ERROR: $*" >&2
  exit 1
}

cleanup() {
  if [ -n "${cid:-}" ]; then
    docker rm -f "$cid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

cd "$REPO_DIR"

# --- Mode selection ---
build_args=()
if [ "$NO_CACHE" = "1" ]; then
  log "Mode: CLEAN (no-cache rebuild)"
  build_args+=(--no-cache)
  if [ "$RESET" = "1" ]; then
    log "Reset compose state (project=$PROJECT)"
    docker compose -p "$PROJECT" down --remove-orphans
  fi
else
  log "Mode: FAST (cached build, no service disruption)"
fi

# --- Deps sync guard ---
if [ "$SKIP_DEPS_SYNC" = "1" ]; then
  log "Deps sync check: SKIPPED (SKIP_DEPS_SYNC=1)"
else
  log "Deps sync check: verifying requirements-docker.txt matches pyproject.toml"
  sync_rc=0
  if python3 "$REPO_DIR/scripts/check_deps_sync.py" 2>/dev/null; then
    log "Deps sync check: OK (host python3)"
  else
    sync_rc=$?
    if [ "$sync_rc" -eq 2 ]; then
      log "Deps sync check: host python3 missing tomllib/tomli, falling back to docker"
      docker run --rm -v "$REPO_DIR":/repo -w /repo python:3.11-slim \
        python scripts/check_deps_sync.py \
        || fail "requirements-docker.txt is out of sync with pyproject.toml"
      log "Deps sync check: OK (docker fallback)"
    else
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
log "Runtime asset checks"
docker exec "$cid" bash -lc '
set -euo pipefail
id
ls -lah /app/yolov8s.pt
stat -c "%u:%g %A %n" /app/.ultralytics /app/yolov8s.pt
test -w /app/.ultralytics
test -s /app/yolov8s.pt
' || fail "runtime asset checks failed"

log "verify_container_assets: ok"
