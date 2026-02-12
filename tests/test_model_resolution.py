"""Tests for model resolution and NO_DETECT mode."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.detector import resolve_model_path
from src.utils import VideoMeta


class TestResolveModelPath:
    """Model resolution checks paths in priority order."""

    def test_tank_model_preferred(self, tmp_path):
        """When tank model exists, it should be selected."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        tank_model = models_dir / "ball_best.pt"
        tank_model.write_bytes(b"model_data")

        config = {
            "paths": {"models": str(models_dir)},
            "model": {"path": "/nonexistent/model.pt"},
        }
        path, mode = resolve_model_path(config)
        assert mode == "normal"
        assert path == str(tank_model)

    def test_config_model_fallback(self, tmp_path):
        """When only config model path exists, it should be selected."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # tank model does NOT exist

        config_model = tmp_path / "config_model.pt"
        config_model.write_bytes(b"model_data")

        config = {
            "paths": {"models": str(models_dir)},
            "model": {"path": str(config_model)},
        }
        path, mode = resolve_model_path(config)
        assert mode == "normal"
        assert path == str(config_model)

    def test_base_model_copied_to_tank(self, tmp_path):
        """When only ball_base.pt exists, it should be copied to tank and selected."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        base_model = Path("/app/models/ball_base.pt")

        config = {
            "paths": {"models": str(models_dir)},
            "model": {"path": "/nonexistent/model.pt"},
        }

        # Mock the base_model.exists() check since /app/models doesn't exist in test
        with patch.object(Path, "exists") as mock_exists:
            original_exists = Path.exists

            def side_effect(self):
                if str(self) == str(models_dir / "ball_best.pt"):
                    return False
                if str(self) == "/nonexistent/model.pt":
                    return False
                if str(self) == "/app/models/ball_base.pt":
                    return True
                return original_exists(self)

            mock_exists.side_effect = side_effect

            with patch("shutil.copy2") as mock_copy:
                path, mode = resolve_model_path(config)
                assert mode == "normal"
                assert str(models_dir / "ball_best.pt") in path
                mock_copy.assert_called_once()

    def test_no_model_returns_no_detect(self, tmp_path):
        """When no model exists, mode should be 'no_detect'."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        config = {
            "paths": {"models": str(models_dir)},
            "model": {"path": str(tmp_path / "nonexistent.pt")},
        }
        path, mode = resolve_model_path(config)
        assert mode == "no_detect"
        assert path is None


class TestCameraStaticPath:
    """Static camera path for NO_DETECT mode."""

    def test_generate_static_constant_path(self, test_config, tmp_path):
        """Static path should have all frames at field center with default FOV."""
        from src.camera import CameraPathGenerator

        camera = CameraPathGenerator(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=10.0,
            duration=1.0, total_frames=10, codec="h264",
        )
        output = tmp_path / "camera_path.json"
        camera.generate_static(meta, output)

        assert output.exists()
        with open(output) as f:
            path = json.load(f)

        assert len(path) == 10
        for entry in path:
            assert entry["yaw"] == test_config["camera"]["field_center_yaw_deg"]
            assert entry["pitch"] == test_config["camera"]["field_center_pitch_deg"]
            assert entry["fov"] == test_config["camera"]["default_fov"]


class TestExporterMode:
    """Exporter includes mode in metadata."""

    def test_metadata_includes_mode(self, test_config, tmp_path):
        """metadata.json should include mode field."""
        from src.exporter import Exporter
        from src.utils import VideoMeta

        # Setup work dir with required outputs
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "broadcast.mp4").write_bytes(b"video")
        (work_dir / "tactical_wide.mp4").write_bytes(b"video")
        (work_dir / "camera_path.json").write_text("[]")

        config = {
            **test_config,
            "paths": {
                **test_config["paths"],
                "processed": str(tmp_path / "processed"),
                "highlights": str(tmp_path / "highlights"),
                "archive_raw": str(tmp_path / "archive"),
            },
        }

        exporter = Exporter(config)
        meta = VideoMeta(
            width=640, height=320, fps=10.0,
            duration=1.0, total_frames=10, codec="h264",
        )
        output_dir = exporter.finalize(
            work_dir, str(tmp_path / "input.mp4"), meta, mode="no_detect"
        )

        metadata = json.loads((output_dir / "metadata.json").read_text())
        assert metadata["mode"] == "no_detect"

    def test_normal_mode_preserves_all_artifacts(self, test_config, tmp_path):
        """Normal mode should try to preserve detection artifacts."""
        from src.exporter import Exporter
        from src.utils import VideoMeta

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "broadcast.mp4").write_bytes(b"video")
        (work_dir / "tactical_wide.mp4").write_bytes(b"video")
        (work_dir / "detections.jsonl").write_text("")
        (work_dir / "tracks.json").write_text("[]")
        (work_dir / "camera_path.json").write_text("[]")
        (work_dir / "hard_frames.json").write_text("{}")

        config = {
            **test_config,
            "paths": {
                **test_config["paths"],
                "processed": str(tmp_path / "processed"),
                "highlights": str(tmp_path / "highlights"),
                "archive_raw": str(tmp_path / "archive"),
            },
        }

        exporter = Exporter(config)
        meta = VideoMeta(
            width=640, height=320, fps=10.0,
            duration=1.0, total_frames=10, codec="h264",
        )
        output_dir = exporter.finalize(
            work_dir, str(tmp_path / "input.mp4"), meta, mode="normal"
        )

        metadata = json.loads((output_dir / "metadata.json").read_text())
        assert metadata["mode"] == "normal"
        # hard_frames.json should be preserved
        assert (output_dir / "hard_frames.json").exists()
