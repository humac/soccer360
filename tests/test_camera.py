"""Tests for camera path generation."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.camera import CameraPathGenerator, angle_diff, unwrap_angles, wrap_angle
from src.utils import VideoMeta, pixel_to_yaw_pitch


# ---------------------------------------------------------------------------
# angle utility tests
# ---------------------------------------------------------------------------

class TestAngleDiff:
    def test_same_angle(self):
        assert angle_diff(0, 0) == 0

    def test_small_positive(self):
        assert abs(angle_diff(10, 5) - 5) < 1e-6

    def test_small_negative(self):
        assert abs(angle_diff(5, 10) - (-5)) < 1e-6

    def test_wrap_positive(self):
        # From 170 to -170 should be +20, not -340
        result = angle_diff(-170, 170)
        assert abs(result - 20) < 1e-6

    def test_wrap_negative(self):
        # From -170 to 170 should be -20, not +340
        result = angle_diff(170, -170)
        assert abs(result - (-20)) < 1e-6

    def test_opposite(self):
        result = angle_diff(180, 0)
        assert abs(abs(result) - 180) < 1e-6


class TestWrapAngle:
    def test_within_range(self):
        assert abs(wrap_angle(90) - 90) < 1e-6

    def test_positive_overflow(self):
        assert abs(wrap_angle(270) - (-90)) < 1e-6

    def test_negative_overflow(self):
        assert abs(wrap_angle(-270) - 90) < 1e-6

    def test_boundary(self):
        assert abs(wrap_angle(360) - 0) < 1e-6


class TestUnwrapAngles:
    def test_no_wrap(self):
        angles = [0, 10, 20, 30]
        result = unwrap_angles(angles)
        assert result == angles

    def test_wrap_around(self):
        # 170, 175, -175, -170 should unwrap to 170, 175, 185, 190
        angles = [170, 175, -175, -170]
        result = unwrap_angles(angles)
        assert abs(result[0] - 170) < 1e-6
        assert abs(result[1] - 175) < 1e-6
        assert abs(result[2] - 185) < 1e-6
        assert abs(result[3] - 190) < 1e-6


# ---------------------------------------------------------------------------
# CameraPathGenerator tests
# ---------------------------------------------------------------------------

class TestPixelToAngle:
    def test_center(self):
        """Center of equirectangular = (0, 0) in yaw/pitch."""
        yaw, pitch = pixel_to_yaw_pitch(160, 80, 320, 160)
        assert abs(yaw) < 1e-6
        assert abs(pitch) < 1e-6

    def test_left_edge(self):
        yaw, pitch = pixel_to_yaw_pitch(0, 80, 320, 160)
        assert abs(yaw - (-180)) < 1e-6

    def test_right_edge(self):
        yaw, pitch = pixel_to_yaw_pitch(320, 80, 320, 160)
        assert abs(yaw - 180) < 1e-6

    def test_top(self):
        yaw, pitch = pixel_to_yaw_pitch(160, 0, 320, 160)
        assert abs(pitch - 90) < 1e-6

    def test_bottom(self):
        yaw, pitch = pixel_to_yaw_pitch(160, 160, 320, 160)
        assert abs(pitch - (-90)) < 1e-6


class TestCameraPathGeneration:
    def test_v1_uses_detection_img_size_for_angle_mapping(self, test_config):
        """V1 camera geometry should use detection.img_size, not legacy detector.resolution."""
        config = deepcopy(test_config)
        config["detection"]["img_size"] = 128  # expected det space: 256x128
        config["detector"]["resolution"] = [320, 160]  # intentionally mismatched legacy value

        gen = CameraPathGenerator(config)
        angles = gen._tracks_to_angles([
            {"frame": 0, "ball": {"x": 128, "y": 64, "confidence": 0.9}},
        ])

        yaw, pitch, _ = angles[0]
        assert gen.det_width == 256
        assert gen.det_height == 128
        assert abs(yaw) < 1e-6
        assert abs(pitch) < 1e-6

    def test_basic_generation(self, test_config, sample_tracks, tmp_work_dir):
        """Camera path should be generated with correct number of entries."""
        gen = CameraPathGenerator(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=1.0, total_frames=30, codec="h264",
        )
        output_path = tmp_work_dir / "camera_path.json"

        gen.generate(sample_tracks, meta, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            path = json.load(f)

        assert len(path) == 30
        for entry in path:
            assert "yaw" in entry
            assert "pitch" in entry
            assert "fov" in entry
            assert -180 <= entry["yaw"] <= 180
            assert -90 <= entry["pitch"] <= 90
            assert test_config["camera"]["min_fov"] <= entry["fov"] <= test_config["camera"]["max_fov"]

    def test_ball_lost_handling(self, test_config, tmp_work_dir):
        """Camera should drift to field center when ball is lost for extended period."""
        gen = CameraPathGenerator(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=5.0, total_frames=150, codec="h264",
        )

        # All frames have ball at yaw=90 for first 10, then lost for 140
        tracks = []
        for i in range(10):
            tracks.append({
                "frame": i,
                "ball": {"x": 240, "y": 80, "confidence": 0.9, "track_id": 1},
            })
        for i in range(10, 150):
            tracks.append({"frame": i, "ball": None})

        tracks_path = tmp_work_dir / "tracks_lost.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        output_path = tmp_work_dir / "camera_path_lost.json"
        gen.generate(tracks_path, meta, output_path)

        with open(output_path) as f:
            path = json.load(f)

        # Last entries should be closer to field center than initial tracking position
        initial_yaw = abs(path[9]["yaw"])
        final_yaw = abs(path[-1]["yaw"])
        assert final_yaw < initial_yaw or abs(final_yaw) < 10  # Should drift toward 0

    def test_smooth_output(self, test_config, sample_tracks, tmp_work_dir):
        """Camera path should be smooth (no large frame-to-frame jumps)."""
        gen = CameraPathGenerator(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=1.0, total_frames=30, codec="h264",
        )
        output_path = tmp_work_dir / "camera_path_smooth.json"
        gen.generate(sample_tracks, meta, output_path)

        with open(output_path) as f:
            path = json.load(f)

        max_delta_per_frame = test_config["camera"]["max_fast_pan_speed_deg_per_sec"] / 30.0

        for i in range(1, len(path)):
            dyaw = abs(angle_diff(path[i]["yaw"], path[i - 1]["yaw"]))
            dpitch = abs(path[i]["pitch"] - path[i - 1]["pitch"])
            assert dyaw <= max_delta_per_frame + 0.1, f"Yaw jump too large at frame {i}: {dyaw}"
            assert dpitch <= max_delta_per_frame + 0.1, f"Pitch jump too large at frame {i}: {dpitch}"
