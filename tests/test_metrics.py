"""Tests for per-phase timing and GPU utilization metrics."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from src.metrics import PhaseTimer, gpu_utilization_snapshot


class TestPhaseTimer:
    def test_records_timing(self):
        """PhaseTimer records wall-clock time for named phases."""
        timer = PhaseTimer()
        with timer.phase("detection"):
            time.sleep(0.01)
        result = timer.to_dict()
        assert "detection" in result["phase_timings_sec"]
        assert result["phase_timings_sec"]["detection"] >= 0.01

    def test_records_multiple_phases(self):
        """PhaseTimer tracks multiple phases independently."""
        timer = PhaseTimer()
        with timer.phase("detection"):
            time.sleep(0.01)
        with timer.phase("tracking"):
            time.sleep(0.01)
        result = timer.to_dict()
        assert len(result["phase_timings_sec"]) == 2
        assert "detection" in result["phase_timings_sec"]
        assert "tracking" in result["phase_timings_sec"]

    def test_records_stats(self):
        """PhaseTimer stores arbitrary stats."""
        timer = PhaseTimer()
        timer.record_stat("detection_count", 42)
        timer.record_stat("track_frames_with_ball", 100)
        result = timer.to_dict()
        assert result["stats"]["detection_count"] == 42
        assert result["stats"]["track_frames_with_ball"] == 100

    def test_empty_timer(self):
        """Empty PhaseTimer returns empty dicts."""
        timer = PhaseTimer()
        result = timer.to_dict()
        assert result == {"phase_timings_sec": {}, "stats": {}}

    def test_stat_accepts_none(self):
        """Stats can store None values (e.g. missing GPU snapshot)."""
        timer = PhaseTimer()
        timer.record_stat("gpu_snapshot", None)
        result = timer.to_dict()
        assert result["stats"]["gpu_snapshot"] is None


class TestGpuUtilizationSnapshot:
    def test_returns_none_when_nvidia_smi_missing(self):
        """GPU snapshot is null-safe when nvidia-smi is absent."""
        with patch("src.metrics.subprocess.run", side_effect=FileNotFoundError):
            assert gpu_utilization_snapshot() is None

    def test_returns_none_on_nonzero_exit(self):
        """GPU snapshot returns None when nvidia-smi fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("src.metrics.subprocess.run", return_value=mock_result):
            assert gpu_utilization_snapshot() is None

    def test_returns_none_on_timeout(self):
        """GPU snapshot returns None on timeout."""
        with patch("src.metrics.subprocess.run", side_effect=TimeoutError):
            assert gpu_utilization_snapshot() is None

    def test_parses_nvidia_smi_output(self):
        """GPU snapshot parses nvidia-smi CSV output correctly."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "45, 30, 2048, 24576, 65\n"
        with patch("src.metrics.subprocess.run", return_value=mock_result):
            snap = gpu_utilization_snapshot()
        assert snap == {
            "gpu_utilization_pct": 45,
            "memory_utilization_pct": 30,
            "memory_used_mb": 2048,
            "memory_total_mb": 24576,
            "temperature_c": 65,
        }

    def test_returns_none_on_malformed_output(self):
        """GPU snapshot returns None if CSV has too few columns."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "45, 30\n"
        with patch("src.metrics.subprocess.run", return_value=mock_result):
            assert gpu_utilization_snapshot() is None
