#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "healthcheck: $1" >&2
  exit 1
}

mountpoint -q /tank || fail "/tank is not mounted"
mountpoint -q /scratch || fail "/scratch is not mounted"

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found"
nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi failed"

mkdir -p /tank/logs || fail "cannot create /tank/logs"
testfile="/tank/logs/.healthcheck.$$"
touch "$testfile" || fail "cannot write to /tank/logs"
rm -f "$testfile"

echo "healthcheck: ok"
