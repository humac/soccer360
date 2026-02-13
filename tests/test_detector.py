"""Tests for the detector module (non-GPU parts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.detector as detector_mod
from src.detector import Detector, resolve_v1_model_path_and_source
from src.utils import VideoMeta, pixel_to_yaw_pitch, wrap_angle_deg


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


class TestV1ModelResolver:
    def test_detector_model_path_wins_over_detection_path(self, tmp_path):
        detector_model = tmp_path / "detector_override.pt"
        detector_model.write_bytes(b"model")
        detection_model = tmp_path / "legacy_detection.pt"
        detection_model.write_bytes(b"model")

        config = {
            "detector": {"model_path": str(detector_model)},
            "detection": {"path": str(detection_model)},
            "mode": {"allow_no_model": True},
        }
        resolved_path, source = resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))
        assert resolved_path == str(detector_model)
        assert source == "detector.model_path"

    def test_detection_path_used_when_detector_model_path_missing(self, tmp_path):
        detection_model = tmp_path / "legacy_detection.pt"
        detection_model.write_bytes(b"model")

        config = {
            "detection": {"path": str(detection_model)},
            "mode": {"allow_no_model": True},
        }
        resolved_path, source = resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))
        assert resolved_path == str(detection_model)
        assert source == "detection.path"

    def test_default_source_used_when_paths_missing(self, tmp_path):
        fine_tuned = tmp_path / "ball_best.pt"
        fine_tuned.write_bytes(b"model")

        config = {
            "mode": {"allow_no_model": True},
        }
        resolved_path, source = resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))
        assert resolved_path == str(fine_tuned)
        assert source == "default"

    def test_explicit_detector_override_bypasses_fine_tuned_preference(self, tmp_path):
        fine_tuned = tmp_path / "ball_best.pt"
        fine_tuned.write_bytes(b"fine_tuned")
        detector_model = tmp_path / "explicit_override.pt"
        detector_model.write_bytes(b"explicit")

        config = {
            "detector": {"model_path": str(detector_model)},
            "mode": {"allow_no_model": True},
        }
        resolved_path, source = resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))
        assert resolved_path == str(detector_model)
        assert source == "detector.model_path"

    def test_missing_explicit_detector_override_raises(self, tmp_path):
        config = {
            "detector": {"model_path": str(tmp_path / "missing_override.pt")},
            "mode": {"allow_no_model": True},
        }

        with pytest.raises(RuntimeError, match="Explicit detector.model_path not found or not a file"):
            resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))

    def test_explicit_detector_override_directory_raises(self, tmp_path):
        override_dir = tmp_path / "override_dir"
        override_dir.mkdir()

        config = {
            "detector": {"model_path": str(override_dir)},
            "mode": {"allow_no_model": True},
        }

        with pytest.raises(RuntimeError, match="Explicit detector.model_path not found or not a file"):
            resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))

    def test_default_detector_model_path_is_not_explicit_override(self, tmp_path):
        fine_tuned = tmp_path / "ball_best.pt"
        fine_tuned.write_bytes(b"fine_tuned")

        config = {
            "detector": {"model_path": "/app/yolov8s.pt"},
            "mode": {"allow_no_model": True},
        }
        resolved_path, source = resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))
        assert resolved_path == str(fine_tuned)
        assert source == "default"


class TestDetectorHardening:
    def test_interpolation_does_not_bridge_large_gaps(self):
        detections = [
            {"frame": 0, "bbox": [0, 0, 10, 10], "confidence": 0.9, "class": 0},
            {"frame": 6, "bbox": [6, 0, 16, 10], "confidence": 0.8, "class": 0},
        ]
        out = Detector._interpolate_skipped(detections, total_frames=7, skip_n=2)
        frames = sorted(d["frame"] for d in out)
        assert frames == [0, 6]

    def test_foi_center_resets_between_runs(self, tmp_path, monkeypatch):
        cfg = {
            "model": {"path": "dummy.pt"},
            "detector": {"resolution": [320, 160], "batch_size": 16},
            "field_of_interest": {
                "enabled": True,
                "center_mode": "auto",
                "center_yaw_deg": 0.0,
                "yaw_window_deg": 120.0,
                "pitch_min_deg": -45.0,
                "pitch_max_deg": 20.0,
                "auto_sample_seconds": 30.0,
                "auto_min_conf": 0.1,
            },
        }
        detector = Detector(cfg)

        def fake_load_model():
            detector._model = object()

        class FakeReader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                frame = np.zeros((160, 320, 3), dtype=np.uint8)
                yield frame
                yield frame

        run_centers = [187.0, 133.0]  # yaw ~ +30, then yaw ~ -30
        run_idx = {"value": 0}

        def fake_detect_batch(frames, indices):
            cx = run_centers[run_idx["value"]]
            run_idx["value"] += 1
            out = []
            for idx in indices:
                out.append({
                    "frame": idx,
                    "bbox": [cx - 5, 75.0, cx + 5, 85.0],
                    "confidence": 0.9,
                    "class": 0,
                })
            return out

        monkeypatch.setattr(detector, "_load_model", fake_load_model)
        monkeypatch.setattr(detector_mod, "FFmpegFrameReader", FakeReader)
        monkeypatch.setattr(detector, "_detect_batch", fake_detect_batch)

        meta = VideoMeta(
            width=640, height=320, fps=10.0,
            duration=0.2, total_frames=2, codec="h264",
        )

        detector.run_streaming("dummy.mp4", meta, tmp_path / "run1" / "detections.jsonl")
        center1 = detector._effective_center_yaw
        detector.run_streaming("dummy.mp4", meta, tmp_path / "run2" / "detections.jsonl")
        center2 = detector._effective_center_yaw

        assert center1 is not None and 20.0 < center1 < 40.0
        assert center2 is not None and -40.0 < center2 < -20.0
