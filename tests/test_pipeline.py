"""Integration tests for the full pipeline."""

from __future__ import annotations

import json
from copy import deepcopy
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

    def test_v1_stabilizer_receives_total_frames_fallback(self, test_config, tmp_path, monkeypatch):
        """V1 pipeline should pass detector/meta-derived total_frames to BallStabilizer."""
        from src.pipeline import Pipeline

        config = deepcopy(test_config)
        config["paths"]["scratch"] = str(tmp_path / "scratch")
        input_path = tmp_path / "input.mp4"
        input_path.write_bytes(b"dummy")

        calls: dict[str, int | None] = {}
        meta = VideoMeta(
            width=640, height=320, fps=30.0, duration=2.0, total_frames=60, codec="h264"
        )

        class _FakeDetector:
            def __init__(self, cfg):
                pass

            def run_streaming(self, video_path, meta_obj, output_path):
                Path(output_path).write_text("")
                return 0  # force fallback to meta.total_frames

        class _FakeStabilizer:
            def __init__(self, cfg):
                pass

            def run(self, detections_path, tracks_path, fps, total_frames=None):
                calls["total_frames"] = total_frames
                Path(tracks_path).write_text("[]")
                return []

        class _FakeActiveLearning:
            def __init__(self, cfg):
                pass

            def run(self, *args, **kwargs):
                return None

        class _FakeCamera:
            def __init__(self, cfg):
                pass

            def generate(self, tracks_path, meta_obj, out_path):
                Path(out_path).write_text("[]")

            def generate_static(self, meta_obj, out_path):
                Path(out_path).write_text("[]")

        class _FakeReframer:
            def __init__(self, cfg):
                pass

            def render_broadcast(self, *args, **kwargs):
                return None

            def render_tactical(self, *args, **kwargs):
                return None

        class _FakeHighlights:
            def __init__(self, cfg):
                pass

            def detect_and_export(self, *args, **kwargs):
                return None

        class _FakeExporter:
            def __init__(self, cfg):
                pass

            def finalize(self, *args, **kwargs):
                return tmp_path / "out"

        monkeypatch.setattr("src.pipeline.resolve_model_path_v1", lambda cfg: ("dummy.pt", "normal"))
        monkeypatch.setattr("src.pipeline.Detector", _FakeDetector)
        monkeypatch.setattr("src.pipeline.BallStabilizer", _FakeStabilizer)
        monkeypatch.setattr("src.pipeline.ActiveLearningExporter", _FakeActiveLearning)
        monkeypatch.setattr("src.pipeline.CameraPathGenerator", _FakeCamera)
        monkeypatch.setattr("src.pipeline.Reframer", _FakeReframer)
        monkeypatch.setattr("src.pipeline.HighlightDetector", _FakeHighlights)
        monkeypatch.setattr("src.pipeline.Exporter", _FakeExporter)
        monkeypatch.setattr("src.pipeline.probe_video", lambda path: meta)

        pipeline = Pipeline(config)
        pipeline.run(input_path, cleanup=False)

        assert calls["total_frames"] == meta.total_frames
