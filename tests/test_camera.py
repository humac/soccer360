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


def _make_cluster_json(tmp_dir: Path, clusters: list[dict]) -> Path:
    """Write a player_cluster.json file and return its path."""
    path = tmp_dir / "player_cluster.json"
    with open(path, "w") as f:
        json.dump(clusters, f)
    return path


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


class TestCenterOfPlayHybrid:
    """Tests for hybrid camera tracking with player cluster blending."""

    def test_ball_lost_follows_cluster(self, test_config, tmp_work_dir):
        """When ball is lost but cluster available, camera follows cluster instead of drifting to center."""
        config = deepcopy(test_config)
        config["center_of_play"]["enabled"] = True

        gen = CameraPathGenerator(config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=5.0, total_frames=150, codec="h264",
        )

        # Ball detected at x=240 for 10 frames, then lost for 140
        tracks = []
        for i in range(10):
            tracks.append({
                "frame": i,
                "ball": {"x": 240, "y": 80, "confidence": 0.9, "track_id": 1},
            })
        for i in range(10, 150):
            tracks.append({"frame": i, "ball": None})

        tracks_path = tmp_work_dir / "tracks_hybrid.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        # Player cluster stays at x=240 (right side) for all 150 frames
        clusters = []
        for i in range(150):
            clusters.append({
                "frame": i,
                "cluster": {"x": 240.0, "y": 80.0, "spread_x_deg": 25.0,
                             "player_count": 15, "confidence": 0.6},
            })
        cluster_path = _make_cluster_json(tmp_work_dir, clusters)

        output_path = tmp_work_dir / "camera_path_hybrid.json"
        gen.generate(tracks_path, meta, output_path, player_cluster_path=cluster_path)

        with open(output_path) as f:
            path = json.load(f)

        # Final yaw should NOT have drifted to field center (0)
        # since cluster is providing a signal at x=240 (positive yaw)
        final_yaw = path[-1]["yaw"]
        assert final_yaw > 10, f"Camera drifted to center despite cluster: yaw={final_yaw}"

    def test_ball_detected_with_slight_cluster_bias(self, test_config, tmp_work_dir):
        """When ball is detected, cluster provides slight bias but ball dominates."""
        config = deepcopy(test_config)
        config["center_of_play"]["enabled"] = True

        gen = CameraPathGenerator(config)

        # Ball at x=200, cluster at x=100 -- should blend slightly toward cluster
        tracks = [{"frame": 0, "ball": {"x": 200, "y": 80, "confidence": 0.9}}]
        clusters = [{"frame": 0, "cluster": {"x": 100, "y": 80, "spread_x_deg": 20.0,
                                              "player_count": 15, "confidence": 0.6}}]

        hybrid_angles = gen._tracks_to_angles_hybrid(tracks, clusters)
        ball_only_angles = gen._tracks_to_angles(tracks)

        hybrid_yaw = hybrid_angles[0][0]
        ball_yaw = ball_only_angles[0][0]

        # Hybrid should be between ball-only and cluster (closer to ball)
        cluster_yaw, _ = pixel_to_yaw_pitch(100, 80, gen.det_width, gen.det_height)
        # With blend=0.15, hybrid should be slightly toward cluster from ball
        assert abs(hybrid_yaw - ball_yaw) > 0.5, "Hybrid should differ from ball-only"
        # But closer to ball than to cluster
        assert abs(hybrid_yaw - ball_yaw) < abs(hybrid_yaw - cluster_yaw)

    def test_disabled_cop_no_effect(self, test_config, tmp_work_dir):
        """With center_of_play disabled, cluster path is ignored."""
        config = deepcopy(test_config)
        config["center_of_play"]["enabled"] = False

        gen = CameraPathGenerator(config)
        assert not gen.cop_enabled

        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=1.0, total_frames=30, codec="h264",
        )

        # Tracks with ball lost
        tracks = [{"frame": i, "ball": None} for i in range(30)]
        tracks_path = tmp_work_dir / "tracks_disabled.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        clusters = [{"frame": i, "cluster": {"x": 240, "y": 80, "spread_x_deg": 25.0,
                                              "player_count": 15, "confidence": 0.6}}
                     for i in range(30)]
        cluster_path = _make_cluster_json(tmp_work_dir, clusters)

        output_path = tmp_work_dir / "camera_path_disabled.json"
        gen.generate(tracks_path, meta, output_path, player_cluster_path=cluster_path)

        # Should use standard ball-only logic (drift to center) since COP disabled
        with open(output_path) as f:
            path = json.load(f)
        # Camera should be at/near field center since all frames are lost
        assert abs(path[-1]["yaw"]) < 15

    def test_fov_widens_with_player_spread(self, test_config, tmp_work_dir):
        """FOV should widen when player spread is large."""
        config = deepcopy(test_config)
        config["center_of_play"]["enabled"] = True
        config["center_of_play"]["fov_from_spread"] = True

        gen = CameraPathGenerator(config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=1.0, total_frames=30, codec="h264",
        )

        # Ball detected, cluster with large spread
        tracks = []
        clusters = []
        for i in range(30):
            tracks.append({
                "frame": i,
                "ball": {"x": 160, "y": 80, "confidence": 0.9},
            })
            clusters.append({
                "frame": i,
                "cluster": {"x": 160, "y": 80, "spread_x_deg": 55.0,
                             "player_count": 20, "confidence": 0.7},
            })

        tracks_path = tmp_work_dir / "tracks_spread.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)
        cluster_path = _make_cluster_json(tmp_work_dir, clusters)

        output_path = tmp_work_dir / "camera_path_spread.json"
        gen.generate(tracks_path, meta, output_path, player_cluster_path=cluster_path)

        with open(output_path) as f:
            path = json.load(f)

        # With spread=55 (near spread_max_deg=60), FOV should be wider than default
        # spread_max_fov is 105.0
        max_fov = max(e["fov"] for e in path)
        assert max_fov > config["camera"]["max_fov"], (
            f"FOV {max_fov} should exceed camera.max_fov={config['camera']['max_fov']} "
            f"due to player spread"
        )


# ---------------------------------------------------------------------------
# Smoothness tests
# ---------------------------------------------------------------------------

class TestFOVSmoothing:
    """Tests for FOV EMA smoothing to prevent zoom jitter."""

    def test_fov_no_frame_to_frame_oscillation(self, test_config, tmp_work_dir):
        """FOV should not oscillate between lost and found states."""
        gen = CameraPathGenerator(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=2.0, total_frames=60, codec="h264",
        )

        # Alternating ball detected / lost every frame (worst case for oscillation)
        tracks = []
        for i in range(60):
            if i % 2 == 0:
                tracks.append({
                    "frame": i,
                    "ball": {"x": 160, "y": 80, "confidence": 0.5},
                })
            else:
                tracks.append({"frame": i, "ball": None})

        tracks_path = tmp_work_dir / "tracks_flicker.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        output_path = tmp_work_dir / "camera_path_flicker.json"
        gen.generate(tracks_path, meta, output_path)

        with open(output_path) as f:
            path = json.load(f)

        # FOV changes between consecutive frames should be small due to EMA
        for i in range(1, len(path)):
            dfov = abs(path[i]["fov"] - path[i - 1]["fov"])
            assert dfov < 3.0, (
                f"FOV jump of {dfov:.1f}° at frame {i} "
                f"({path[i-1]['fov']} -> {path[i]['fov']})"
            )

    def test_fov_gradual_transition_on_ball_loss(self, test_config, tmp_work_dir):
        """FOV should widen gradually when ball is lost, not snap immediately."""
        gen = CameraPathGenerator(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=2.0, total_frames=60, codec="h264",
        )

        # Ball detected for 15 frames, then lost for 45
        tracks = []
        for i in range(15):
            tracks.append({
                "frame": i,
                "ball": {"x": 160, "y": 80, "confidence": 0.9},
            })
        for i in range(15, 60):
            tracks.append({"frame": i, "ball": None})

        tracks_path = tmp_work_dir / "tracks_gradual.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        output_path = tmp_work_dir / "camera_path_gradual.json"
        gen.generate(tracks_path, meta, output_path)

        with open(output_path) as f:
            path = json.load(f)

        # Frame 15 is first lost frame; FOV should not snap to max_fov
        fov_at_loss = path[15]["fov"]
        max_fov = test_config["camera"]["max_fov"]
        assert fov_at_loss < max_fov, (
            f"FOV snapped to {fov_at_loss} on first lost frame "
            f"(max_fov={max_fov}), should transition gradually"
        )

        # But eventually it should approach max_fov
        fov_late = path[-1]["fov"]
        assert fov_late > max_fov - 2.0, (
            f"FOV should approach max_fov after sustained loss, got {fov_late}"
        )


class TestSpreadCarryforward:
    """Tests for spread data carryforward across cluster gaps."""

    def test_spread_gap_no_fov_drop(self, test_config, tmp_work_dir):
        """FOV should not drop when cluster data has a 1-frame gap."""
        config = deepcopy(test_config)
        config["center_of_play"]["enabled"] = True
        config["center_of_play"]["fov_from_spread"] = True

        gen = CameraPathGenerator(config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=1.0, total_frames=30, codec="h264",
        )

        tracks = []
        clusters = []
        for i in range(30):
            tracks.append({
                "frame": i,
                "ball": {"x": 160, "y": 80, "confidence": 0.9},
            })
            # Gap at frames 15 and 16 (no cluster)
            if i not in (15, 16):
                clusters.append({
                    "frame": i,
                    "cluster": {"x": 160, "y": 80, "spread_x_deg": 50.0,
                                 "player_count": 18, "confidence": 0.7},
                })

        tracks_path = tmp_work_dir / "tracks_gap.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)
        cluster_path = _make_cluster_json(tmp_work_dir, clusters)

        output_path = tmp_work_dir / "camera_path_gap.json"
        gen.generate(tracks_path, meta, output_path, player_cluster_path=cluster_path)

        with open(output_path) as f:
            path = json.load(f)

        # FOV at gap frames should not drop significantly vs. neighbors
        fov_14 = path[14]["fov"]
        fov_15 = path[15]["fov"]
        fov_16 = path[16]["fov"]
        fov_17 = path[17]["fov"]
        assert abs(fov_15 - fov_14) < 2.0, (
            f"FOV dropped at gap frame 15: {fov_14} -> {fov_15}"
        )
        assert abs(fov_16 - fov_15) < 2.0, (
            f"FOV dropped at gap frame 16: {fov_15} -> {fov_16}"
        )


class TestSmoothSpeedThresholds:
    """Tests for smooth (non-binary) speed threshold transitions."""

    def test_no_pan_speed_flicker(self, test_config, tmp_work_dir):
        """Pan speed limit should transition smoothly, not flicker."""
        gen = CameraPathGenerator(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=2.0, total_frames=60, codec="h264",
        )

        # Ball moving at moderate speed (should be near the threshold zone)
        tracks = []
        for i in range(60):
            x = 160 + i * 2  # gradual rightward movement
            tracks.append({
                "frame": i,
                "ball": {"x": min(x, 310), "y": 80, "confidence": 0.8},
            })

        tracks_path = tmp_work_dir / "tracks_moderate.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        output_path = tmp_work_dir / "camera_path_moderate.json"
        gen.generate(tracks_path, meta, output_path)

        with open(output_path) as f:
            path = json.load(f)

        # Check that yaw changes are smooth (no sudden speed reversals)
        dyaws = [
            angle_diff(path[i]["yaw"], path[i - 1]["yaw"])
            for i in range(2, len(path))
        ]
        # Consecutive dyaw changes should not reverse sign abruptly by large amounts
        for i in range(1, len(dyaws)):
            accel = abs(dyaws[i] - dyaws[i - 1])
            assert accel < 2.0, (
                f"Camera acceleration spike at frame {i+2}: "
                f"dyaw went {dyaws[i-1]:.2f} -> {dyaws[i]:.2f}"
            )
