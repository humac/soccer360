#!/usr/bin/env bash
set -euo pipefail

# Restart-loop smoke test for watcher dedupe persistence.
# Runs three cases:
#   A) archive_mode=copy
#   B) archive_mode=leave
#   C) archive_collision=skip with deterministic template and exact precreated target
#
# Usage:
#   scripts/smoke_dedupe_restart.sh /path/to/sample.mp4
#
# Optional overrides:
#   INGEST, PROCESSED, ARCHIVE, CONFIG_FILE, BASENAME

INGEST="${INGEST:-/tank/ingest}"
PROCESSED="${PROCESSED:-/tank/processed}"
ARCHIVE="${ARCHIVE:-/tank/archive_raw}"
CONFIG_FILE="${CONFIG_FILE:-configs/pipeline.yaml}"

SAMPLE="${1:-${SAMPLE:-}}"
if [[ -z "${SAMPLE}" ]]; then
  echo "Usage: $0 /path/to/sample.mp4"
  exit 1
fi
if [[ ! -f "${SAMPLE}" ]]; then
  echo "Sample file not found: ${SAMPLE}"
  exit 1
fi

BASENAME="${BASENAME:-$(basename "${SAMPLE}")}"
STATE="${PROCESSED}/.state/watcher_processed_ingest.json"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}"
  exit 1
fi

CONFIG_BACKUP="$(mktemp)"
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
restore_config() {
  cp "${CONFIG_BACKUP}" "${CONFIG_FILE}" || true
  rm -f "${CONFIG_BACKUP}" || true
}
trap restore_config EXIT

set_ingest_config() {
  local archive_mode="$1"
  local archive_collision="$2"
  local archive_template="$3"

  sed -i -E "s|^  archive_mode: .*|  archive_mode: ${archive_mode}|" "${CONFIG_FILE}"
  sed -i -E "s|^  archive_collision: .*|  archive_collision: ${archive_collision}|" "${CONFIG_FILE}"
  sed -i -E "s|^  archive_name_template: .*|  archive_name_template: \"${archive_template}\"|" "${CONFIG_FILE}"
}

clean_all() {
  docker compose down -v || true
  rm -rf "${INGEST:?}/"* "${ARCHIVE:?}/"* "${PROCESSED:?}/"*
  rm -rf "${PROCESSED:?}/.state"
  mkdir -p "${INGEST}" "${PROCESSED}" "${ARCHIVE}" "${PROCESSED}/.state"
}

wait_for_metadata() {
  timeout 300 bash -lc \
    'until find /tank/processed -name metadata.json | grep -q .; do sleep 1; done'
}

state_key_uniqueness_check() {
  python - <<'PY'
import json
import os
import sys

state_path = os.environ["STATE"]
ingest_path = os.environ["INGEST_PATH"]
ingest_real = os.path.realpath(ingest_path)

def normalize_fp(fp):
    if not isinstance(fp, dict):
        return None
    if "size" not in fp or "mtime_ns" not in fp:
        return None
    material = {"size": fp["size"], "mtime_ns": fp["mtime_ns"]}
    ino = fp.get("ino")
    dev = fp.get("dev")
    if isinstance(ino, int) and ino > 0:
        material["ino"] = ino
    if isinstance(dev, int) and dev > 0:
        material["dev"] = dev
    return json.dumps(material, sort_keys=True, separators=(",", ":"))

with open(state_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
    print("FAIL: unexpected state format")
    sys.exit(1)

entries = data["entries"]
entry = entries.get(ingest_real) or entries.get(ingest_path)
if not isinstance(entry, dict):
    print("FAIL: no state entry for ingest path", ingest_path)
    sys.exit(1)

if not entry.get("processed_at"):
    print("FAIL: processed_at missing for", ingest_path)
    sys.exit(1)

target_key = normalize_fp(entry.get("fingerprint"))
if not target_key:
    print("FAIL: missing or invalid fingerprint in state entry")
    sys.exit(1)

matches = 0
for key, record in entries.items():
    if not isinstance(record, dict):
        continue
    if key not in (ingest_path, ingest_real):
        continue
    if normalize_fp(record.get("fingerprint")) == target_key:
        if not record.get("processed_at"):
            print("FAIL: processed_at missing on matching entry")
            sys.exit(1)
        matches += 1

if matches != 1:
    print("FAIL: expected exactly 1 completion record for (path,fingerprint_key), got", matches)
    sys.exit(1)

print("OK: state uniqueness + processed_at check passed")
PY
}

baseline_counts() {
  local jobdirs files
  jobdirs="$(find "${PROCESSED}" -mindepth 1 -maxdepth 2 -type d \
    ! -path "${PROCESSED}/.state*" | wc -l | tr -d ' ')"
  files="$(find "${PROCESSED}" -type f ! -path "${PROCESSED}/.state/*" \
    | wc -l | tr -d ' ')"
  echo "${jobdirs} ${files}"
}

checksum_state() {
  if [[ ! -f "${STATE}" ]]; then
    echo "MISSING"
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${STATE}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${STATE}" | awk '{print $1}'
  else
    python - <<'PY'
import hashlib
import os
state = os.environ["STATE"]
with open(state, "rb") as f:
    print(hashlib.sha256(f.read()).hexdigest())
PY
  fi
}

run_case() {
  local case_name="$1"
  local mode="$2"
  local collision="$3"
  local template="$4"
  local setup_collision="$5"

  echo "=== CASE: ${case_name} ==="
  set_ingest_config "${mode}" "${collision}" "${template}"
  clean_all

  if [[ "${setup_collision}" == "yes" ]]; then
    : > "${ARCHIVE}/${BASENAME}"
  fi

  docker compose up -d worker

  docker compose logs --tail=200 worker \
    | grep -E "ingest_dedupe_state .*processed_state_file=.* persistence=" >/dev/null

  cp "${SAMPLE}" "${INGEST}/${BASENAME}.part"
  sync
  mv "${INGEST}/${BASENAME}.part" "${INGEST}/${BASENAME}"

  wait_for_metadata

  test -f "${STATE}"
  export STATE
  export INGEST_PATH="${INGEST}/${BASENAME}"
  state_key_uniqueness_check

  local base_counts base_state_sum new_counts new_state_sum
  base_counts="$(baseline_counts)"
  base_state_sum="$(checksum_state)"

  docker compose restart worker
  sleep 3

  new_counts="$(baseline_counts)"
  new_state_sum="$(checksum_state)"
  test "${new_counts}" = "${base_counts}"
  test "${new_state_sum}" = "${base_state_sum}"

  docker compose logs worker \
    | grep -E "Skipping already processed ingest file|dedupe skip|skip.*processed" >/dev/null

  echo "PASS: ${case_name}"
}

run_case "A: archive_mode=copy" "copy" "suffix" "{match}_{job_id}{ext}" "no"
run_case "B: archive_mode=leave" "leave" "suffix" "{match}_{job_id}{ext}" "no"
run_case "C: archive_collision=skip deterministic template" "move" "skip" "{match}{ext}" "yes"

echo "All restart-loop smoke cases passed."
