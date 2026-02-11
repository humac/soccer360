"""Tests for the detector module (non-GPU parts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.detector import Detector
from src.utils import pixel_to_yaw_pitch, wrap_angle_deg


# ---------------------------------------------------------------------------
# wrap_angle_deg tests
# ---------------------------------------------------------------------------

class TestWrapAngleDeg:
    def test_identity(self):
        assert wrap_angle_deg(0.0) == 0.0
        assert wrap_angle_deg(90.0) == 90.0
        assert wrap_angle_deg(-90.0) == -90.0

    def test_positive_overflow(self):
        assert abs(wrap_angle_deg(270.0) - (-90.0)) < 1e-9
        assert abs(wrap_angle_deg(450.0) - 90.0) < 1e-9

    def test_negative(self):
        assert abs(wrap_angle_deg(-270.0) - 90.0) < 1e-9

    def test_360_wraps_to_zero(self):
        # 360 % 360 = 0.0, which is in (-180, 180]
        assert abs(wrap_angle_deg(360.0)) < 1e-9

    def test_near_seam(self):
        # +180 should stay as +180 (in (-180, 180])
        assert abs(wrap_angle_deg(180.0) - 180.0) < 1e-9
        # -180 should map to +180
        assert abs(wrap_angle_deg(-180.0) - 180.0) < 1e-9
        # Just inside the seam
        assert abs(wrap_angle_deg(179.9) - 179.9) < 1e-9
        assert abs(wrap_angle_deg(-179.9) - (-179.9)) < 1e-9


# ---------------------------------------------------------------------------
# pixel_to_yaw_pitch tests (basic sanity, detailed tests in test_camera.py)
# ---------------------------------------------------------------------------

class TestPixelToYawPitch:
    def test_center(self):
        yaw, pitch = pixel_to_yaw_pitch(160, 80, 320, 160)
        assert abs(yaw) < 1e-6
        assert abs(pitch) < 1e-6

    def test_edges(self):
        yaw_left, _ = pixel_to_yaw_pitch(0, 80, 320, 160)
        assert abs(yaw_left - (-180.0)) < 1e-6
        yaw_right, _ = pixel_to_yaw_pitch(320, 80, 320, 160)
        assert abs(yaw_right - 180.0) < 1e-6


# ---------------------------------------------------------------------------
# Field-of-Interest filter tests
# ---------------------------------------------------------------------------

def _make_detector(foi_overrides: dict | None = None) -> Detector:
    """Create a Detector with FoI config but no YOLO model loading."""
    foi_cfg = {
        "enabled": True,
        "center_mode": "fixed",
        "center_yaw_deg": 0,
        "yaw_window_deg": 200,
        "pitch_min_deg": -45,
        "pitch_max_deg": 20,
        "auto_sample_seconds": 30,
        "auto_min_conf": 0.25,
    }
    if foi_overrides:
        foi_cfg.update(foi_overrides)

    config = {
        "model": {"path": "dummy.pt"},
        "detector": {"resolution": [320, 160]},
        "field_of_interest": foi_cfg,
    }
    return Detector(config)


def _make_det(frame: int, cx: float, cy: float, conf: float = 0.9) -> dict:
    """Create a detection dict with bbox centered at (cx, cy)."""
    return {
        "frame": frame,
        "bbox": [cx - 5, cy - 5, cx + 5, cy + 5],
        "confidence": conf,
        "class": 0,
    }


class TestFieldOfInterestFilter:
    """Test FoI filtering with 320x160 equirectangular detection space.

    Yaw mapping: x=0 -> -180, x=160 -> 0, x=320 -> +180
    Pitch mapping: y=0 -> +90, y=80 -> 0, y=160 -> -90
    """

    def test_front_detections_survive(self, tmp_path):
        """Detections at frame center (yaw~0) should pass."""
        det = _make_detector()
        dets = [_make_det(0, 160, 80)]  # center: yaw=0, pitch=0
        result = det._filter_foi(dets, 320, 160, 10.0, tmp_path)
        assert len(result) == 1

    def test_rear_detections_filtered(self, tmp_path):
        """Detections at frame edges (yaw~±170) should be rejected."""
        det = _make_detector()
        dets = [
            _make_det(0, 5, 80),    # yaw ~ -174
            _make_det(0, 315, 80),   # yaw ~ +174
        ]
        result = det._filter_foi(dets, 320, 160, 10.0, tmp_path)
        assert len(result) == 0

    def test_pitch_filtering(self, tmp_path):
        """Detections outside pitch range should be rejected."""
        det = _make_detector()
        dets = [
            _make_det(0, 160, 5),    # pitch ~ +84 (above max 20)
            _make_det(0, 160, 155),   # pitch ~ -84 (below min -45)
            _make_det(0, 160, 80),    # pitch = 0 (in range)
        ]
        result = det._filter_foi(dets, 320, 160, 10.0, tmp_path)
        assert len(result) == 1

    def test_disabled_passes_all(self, tmp_path):
        """FoI disabled should return all detections unchanged."""
        det = _make_detector({"enabled": False})
        dets = [
            _make_det(0, 5, 80),
            _make_det(0, 315, 80),
            _make_det(0, 160, 5),
        ]
        result = det._filter_foi(dets, 320, 160, 10.0, tmp_path)
        assert len(result) == 3

    def test_auto_center_mode(self, tmp_path):
        """Auto-center should lock to dominant yaw cluster."""
        det = _make_detector({
            "center_mode": "auto",
            "yaw_window_deg": 60,  # narrow window: +/-30 from auto center
        })
        # Sample detections clustered at yaw ~ +30 (x ~ 176 in 320px)
        # x = (yaw + 180) / 360 * 320 -> yaw=30: x = 210/360 * 320 ~ 186.7
        dets = [
            _make_det(0, 187, 80, 0.9),  # yaw ~ +30
            _make_det(1, 185, 80, 0.8),  # yaw ~ +28
            _make_det(2, 190, 80, 0.85), # yaw ~ +33
            _make_det(3, 10, 80, 0.9),   # yaw ~ -169 (rear, should be filtered)
        ]
        result = det._filter_foi(dets, 320, 160, 10.0, tmp_path)
        # Auto should center near yaw~+30, so only rear det is outside +/-30
        assert len(result) == 3
        assert det._effective_center_yaw is not None
        assert 20.0 < det._effective_center_yaw < 40.0

    def test_auto_center_no_samples_fallback(self, tmp_path):
        """Auto-center with no valid samples falls back to configured center."""
        det = _make_detector({
            "center_mode": "auto",
            "auto_min_conf": 0.99,  # very high threshold -> no samples
        })
        dets = [_make_det(0, 160, 80, 0.5)]
        det._filter_foi(dets, 320, 160, 10.0, tmp_path)
        assert det._effective_center_yaw == 0.0  # fallback to configured

    def test_foi_meta_written(self, tmp_path):
        """foi_meta.json should be written to work_dir."""
        import json

        det = _make_detector()
        dets = [_make_det(0, 160, 80)]
        det._filter_foi(dets, 320, 160, 10.0, tmp_path)

        meta_path = tmp_path / "foi_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["enabled"] is True
        assert meta["center_mode"] == "fixed"
        assert meta["effective_center_yaw_deg"] == 0.0


class TestNMS:
    def test_no_overlap(self):
        boxes = np.array([[0, 0, 5, 5], [10, 10, 15, 15]])
        scores = np.array([0.9, 0.8])
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2

    def test_full_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]])
        scores = np.array([0.9, 0.8])
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 0  # highest score kept

    def test_partial_overlap_below_threshold(self):
        boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20]])
        scores = np.array([0.9, 0.8])
        # IoU = 25/275 ~ 0.09, below 0.5 threshold
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2

    def test_empty(self):
        boxes = np.empty((0, 4))
        scores = np.empty(0)
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 0

    def test_single_box(self):
        boxes = np.array([[0, 0, 10, 10]])
        scores = np.array([0.9])
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert keep == [0]
