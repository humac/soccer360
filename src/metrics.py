"""Per-phase timing and quality metrics for pipeline runs."""

from __future__ import annotations

import logging
import subprocess
import time
from contextlib import contextmanager

logger = logging.getLogger("soccer360.metrics")


class PhaseTimer:
    """Collects per-phase wall-clock timings and optional stats."""

    def __init__(self):
        self._timings: dict[str, float] = {}
        self._stats: dict[str, object] = {}

    @contextmanager
    def phase(self, name: str):
        """Context manager that records wall-clock seconds for a named phase."""
        start = time.monotonic()
        yield
        elapsed = time.monotonic() - start
        self._timings[name] = round(elapsed, 3)
        logger.info("Phase '%s' completed in %.3fs", name, elapsed)

    def record_stat(self, key: str, value: object):
        """Record an arbitrary metric (detection count, track count, etc.)."""
        self._stats[key] = value

    def to_dict(self) -> dict:
        """Return serializable metrics dict."""
        return {
            "phase_timings_sec": dict(self._timings),
            "stats": dict(self._stats),
        }


def gpu_utilization_snapshot() -> dict | None:
    """Capture GPU utilization via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,"
                "memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split("\n")
        if not lines:
            return None
        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) < 5:
            return None
        return {
            "gpu_utilization_pct": int(parts[0]),
            "memory_utilization_pct": int(parts[1]),
            "memory_used_mb": int(parts[2]),
            "memory_total_mb": int(parts[3]),
            "temperature_c": int(parts[4]),
        }
    except Exception:
        return None
