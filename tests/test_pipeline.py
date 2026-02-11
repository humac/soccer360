"""Integration tests for the full pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils import VideoMeta, probe_video


class TestProbeVideo:
    def test_probe_synthetic(self, synthetic_video):
        meta = probe_video(synthetic_video)
        assert meta.width == 640
        assert meta.height == 320
        assert meta.fps > 0
        assert meta.duration > 0
        assert meta.total_frames > 0
        assert meta.codec != "unknown"


class TestPipelineIntegration:
    """Integration test with mocked YOLO detector.

    Tests the full pipeline from detection output through to final export,
    skipping the actual YOLO inference which requires a GPU and model weights.
    """

    def test_tracking_through_export(self, test_config, tmp_path, synthetic_video):
        """Test phases 2-7 of the pipeline using pre-generated detections."""
        from src.camera import CameraPathGenerator
        from src.tracker import Tracker

        # Setup working directory
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        meta = probe_video(synthetic_video)

        # Create synthetic detections (ball moving left to right)
        detections_path = work_dir / "detections.jsonl"
        det_w, det_h = test_config["detector"]["resolution"]

        with open(detections_path, "w") as f:
            for frame in range(meta.total_frames):
                x = int((frame / meta.total_frames) * det_w * 0.8 + det_w * 0.1)
                y = det_h // 2
                det = {
                    "frame": frame,
                    "bbox": [x - 3, y - 3, x + 3, y + 3],
                    "confidence": 0.9,
                    "class": 0,
                }
                f.write(json.dumps(det) + "\n")

        # Phase 2: Tracking
        tracker = Tracker(test_config)
        tracks_path = work_dir / "tracks.json"
        tracker.run(detections_path, tracks_path)

        assert tracks_path.exists()
        with open(tracks_path) as f:
            tracks = json.load(f)
        assert len(tracks) == meta.total_frames

        # Phase 3: Camera path
        camera = CameraPathGenerator(test_config)
        camera_path_file = work_dir / "camera_path.json"
        camera.generate(tracks_path, meta, camera_path_file)

        assert camera_path_file.exists()
        with open(camera_path_file) as f:
            cam_path = json.load(f)
        assert len(cam_path) == meta.total_frames

        # Verify camera path entries are valid
        for entry in cam_path:
            assert -180 <= entry["yaw"] <= 180
            assert -90 <= entry["pitch"] <= 90
            assert entry["fov"] > 0
