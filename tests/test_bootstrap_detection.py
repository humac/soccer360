"""Tests for V1 bootstrap detection: model resolution, y-range filter, best-per-frame."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.detector import Detector, resolve_model_path_v1


# ---------------------------------------------------------------------------
# resolve_model_path_v1 tests (all use tmp_path, no /app dependency)
# ---------------------------------------------------------------------------

class TestResolveModelPathV1:
    def test_fine_tuned_preferred(self, tmp_path):
        """When ball_best.pt exists in models_dir, it should be selected."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "ball_best.pt").write_bytes(b"model_data")
        base = tmp_path / "yolov8s.pt"
        base.write_bytes(b"base_data")

        config = {"detection": {"path": "yolov8s.pt"}, "mode": {"allow_no_model": True}}
        path, mode = resolve_model_path_v1(
            config, models_dir=str(models_dir), base_model_path=str(base)
        )
        assert mode == "normal"
        assert path == str(models_dir / "ball_best.pt")

    def test_baked_fallback(self, tmp_path):
        """When no ball_best.pt, baked yolov8s.pt should be used."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        base = tmp_path / "yolov8s.pt"
        base.write_bytes(b"base_data")

        config = {"detection": {"path": "yolov8s.pt"}, "mode": {"allow_no_model": True}}
        path, mode = resolve_model_path_v1(
            config, models_dir=str(models_dir), base_model_path=str(base)
        )
        assert mode == "normal"
        assert path == str(base)

    def test_no_detect_when_allowed(self, tmp_path):
        """When nothing exists and allow_no_model=true, return NO_DETECT."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        config = {"detection": {"path": "yolov8s.pt"}, "mode": {"allow_no_model": True}}
        path, mode = resolve_model_path_v1(
            config,
            models_dir=str(models_dir),
            base_model_path=str(tmp_path / "nonexistent.pt"),
        )
        assert mode == "no_detect"
        assert path is None

    def test_raises_when_not_allowed(self, tmp_path):
        """When nothing exists and allow_no_model=false, raise RuntimeError."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        config = {"detection": {"path": "yolov8s.pt"}, "mode": {"allow_no_model": False}}
        with pytest.raises(RuntimeError):
            resolve_model_path_v1(
                config,
                models_dir=str(models_dir),
                base_model_path=str(tmp_path / "nonexistent.pt"),
            )

    def test_explicit_path_override(self, tmp_path):
        """User sets non-default detection.path → that path used if exists."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "ball_best.pt").write_bytes(b"finetuned")
        custom = tmp_path / "custom_model.pt"
        custom.write_bytes(b"custom_data")

        config = {
            "detection": {"path": str(custom)},
            "mode": {"allow_no_model": True},
        }
        path, mode = resolve_model_path_v1(
            config, models_dir=str(models_dir), base_model_path=str(tmp_path / "y.pt")
        )
        assert mode == "normal"
        assert path == str(custom)


# ---------------------------------------------------------------------------
# Y-range filter tests
# ---------------------------------------------------------------------------

class TestYRangeFilter:
    """Test vertical band filtering on detections.

    Height=100 for simple fraction computation.
    min_y_frac=0.20, max_y_frac=0.98 (default config).
    """

    @staticmethod
    def _make_det(frame: int, cy: float) -> dict:
        return {
            "frame": frame,
            "bbox": [50, cy - 5, 60, cy + 5],
            "confidence": 0.8,
            "class": 0,
        }

    def test_center_passes(self):
        """Detection at y_frac=0.5 should survive."""
        dets = [self._make_det(0, 50.0)]  # cy=50, y_frac=0.5
        result = Detector._filter_y_range(dets, 100, 0.20, 0.98)
        assert len(result) == 1

    def test_sky_filtered(self):
        """Detection at y_frac=0.05 (sky) should be rejected."""
        dets = [self._make_det(0, 5.0)]  # cy=5, y_frac=0.05
        result = Detector._filter_y_range(dets, 100, 0.20, 0.98)
        assert len(result) == 0

    def test_bottom_filtered(self):
        """Detection at y_frac=0.99 (bottom) should be rejected."""
        dets = [self._make_det(0, 99.0)]  # cy=99, y_frac=0.99
        result = Detector._filter_y_range(dets, 100, 0.20, 0.98)
        assert len(result) == 0

    def test_boundary_inclusive(self):
        """Detections at exactly min and max should pass."""
        dets = [
            self._make_det(0, 20.0),  # y_frac=0.20 (exact min)
            self._make_det(1, 98.0),  # y_frac=0.98 (exact max)
        ]
        result = Detector._filter_y_range(dets, 100, 0.20, 0.98)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Best-per-frame selection tests
# ---------------------------------------------------------------------------

class TestBestPerFrameSelection:
    def test_picks_highest(self):
        """Three detections on same frame → highest confidence kept."""
        dets = [
            {"frame": 0, "bbox": [0, 0, 10, 10], "confidence": 0.5, "class": 0},
            {"frame": 0, "bbox": [20, 20, 30, 30], "confidence": 0.9, "class": 0},
            {"frame": 0, "bbox": [40, 40, 50, 50], "confidence": 0.3, "class": 0},
        ]
        result = Detector._select_best_per_frame(dets)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_one_per_frame(self):
        """Multiple frames each get exactly one detection."""
        dets = [
            {"frame": 0, "bbox": [0, 0, 10, 10], "confidence": 0.5, "class": 0},
            {"frame": 0, "bbox": [20, 20, 30, 30], "confidence": 0.9, "class": 0},
            {"frame": 1, "bbox": [0, 0, 10, 10], "confidence": 0.7, "class": 0},
            {"frame": 1, "bbox": [20, 20, 30, 30], "confidence": 0.4, "class": 0},
            {"frame": 2, "bbox": [0, 0, 10, 10], "confidence": 0.6, "class": 0},
        ]
        result = Detector._select_best_per_frame(dets)
        assert len(result) == 3
        # Verify each frame kept the best
        by_frame = {d["frame"]: d for d in result}
        assert by_frame[0]["confidence"] == 0.9
        assert by_frame[1]["confidence"] == 0.7
        assert by_frame[2]["confidence"] == 0.6

    def test_empty(self):
        """No detections → empty list."""
        result = Detector._select_best_per_frame([])
        assert result == []

    def test_single_detection(self):
        """Single detection passes through."""
        dets = [{"frame": 5, "bbox": [0, 0, 10, 10], "confidence": 0.8, "class": 0}]
        result = Detector._select_best_per_frame(dets)
        assert len(result) == 1
        assert result[0]["frame"] == 5
