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

    def test_low_confidence_ball_caps_cluster_influence(self, test_config):
        """Low-confidence ball measurements should not jump to a 50/50 cluster blend."""
        config = deepcopy(test_config)
        config["center_of_play"]["enabled"] = True
        config["center_of_play"]["ball_blend_weight"] = 0.05
        config["center_of_play"]["low_conf_ball_blend_weight"] = 0.20

        gen = CameraPathGenerator(config)

        tracks = [{"frame": 0, "ball": {"x": 220, "y": 80, "confidence": 0.2}}]
        clusters = [{"frame": 0, "cluster": {"x": 80, "y": 80, "spread_x_deg": 20.0,
                                              "player_count": 12, "confidence": 0.7}}]

        hybrid_angles = gen._tracks_to_angles_hybrid(tracks, clusters)
        ball_only_angles = gen._tracks_to_angles(tracks)

        hybrid_yaw = hybrid_angles[0][0]
        ball_yaw = ball_only_angles[0][0]
        cluster_yaw, _ = pixel_to_yaw_pitch(80, 80, gen.det_width, gen.det_height)

        # The low-confidence hybrid result should still remain notably closer to the ball
        # than to the cluster-only signal.
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
            # Allow up to deadband-sized acceleration spikes (deadband can
            # cause 0 -> deadband_deg transitions).
            limit = test_config["camera"]["deadband_deg"] + 0.5
            assert accel <= limit, (
                f"Camera acceleration spike at frame {i+2}: "
                f"dyaw went {dyaws[i-1]:.2f} -> {dyaws[i]:.2f}"
            )


# ---------------------------------------------------------------------------
# Spatial dead-zone tests
# ---------------------------------------------------------------------------

class TestSpatialDeadzone:
    """Tests for the spatial dead-zone camera feature."""

    def _make_gen(self, test_config, enabled=True, frac=0.30, ramp=0.20):
        config = deepcopy(test_config)
        config["camera"]["spatial_deadzone_enabled"] = enabled
        config["camera"]["spatial_deadzone_frac"] = frac
        config["camera"]["spatial_deadzone_ramp"] = ramp
        return CameraPathGenerator(config)

    def test_no_pan_when_ball_centered(self, test_config, tmp_work_dir):
        """Ball sitting in the center of the frame should produce no camera pan."""
        gen = self._make_gen(test_config, enabled=True, frac=0.30, ramp=0.20)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=1.0, total_frames=30, codec="h264",
        )

        # Ball sits exactly at frame center for all frames
        tracks = [
            {"frame": i, "ball": {"x": 160, "y": 80, "confidence": 0.9}}
            for i in range(30)
        ]
        tracks_path = tmp_work_dir / "tracks_dz_center.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        output_path = tmp_work_dir / "cam_dz_center.json"
        gen.generate(tracks_path, meta, output_path)

        with open(output_path) as f:
            path = json.load(f)

        # After the camera settles, yaw changes should be near zero
        for i in range(5, len(path)):
            dyaw = abs(angle_diff(path[i]["yaw"], path[i - 1]["yaw"]))
            assert dyaw < 0.5, f"Camera panned {dyaw:.2f}° at frame {i} with ball centered"

    def test_pan_ramps_as_ball_approaches_edge(self, test_config, tmp_work_dir):
        """Camera gain should increase smoothly as ball moves toward frame edge."""
        gen = self._make_gen(test_config, enabled=True, frac=0.30, ramp=0.20)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=2.0, total_frames=60, codec="h264",
        )

        # Ball moves slowly from center toward the right edge
        tracks = []
        for i in range(60):
            x = 160 + i * 2.5  # center to right edge
            tracks.append({
                "frame": i,
                "ball": {"x": min(x, 310), "y": 80, "confidence": 0.9},
            })
        tracks_path = tmp_work_dir / "tracks_dz_ramp.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        output_path = tmp_work_dir / "cam_dz_ramp.json"
        gen.generate(tracks_path, meta, output_path)

        with open(output_path) as f:
            path = json.load(f)

        # Camera should start still (ball in deadzone) then gradually speed up.
        # Compare average speed (not max) because the deadband can cause
        # initial spikes when movement first crosses the threshold.
        early_dyaws = [
            abs(angle_diff(path[i]["yaw"], path[i - 1]["yaw"]))
            for i in range(3, 15)
        ]
        late_dyaws = [
            abs(angle_diff(path[i]["yaw"], path[i - 1]["yaw"]))
            for i in range(40, 55)
        ]
        assert sum(late_dyaws) / len(late_dyaws) > sum(early_dyaws) / len(early_dyaws), (
            "Camera should pan faster on average as ball approaches edge"
        )

    def test_disabled_by_default(self, test_config, sample_tracks, tmp_work_dir):
        """With spatial_deadzone_enabled=False, output should match the non-deadzone path."""
        gen_off = self._make_gen(test_config, enabled=False)
        gen_baseline = CameraPathGenerator(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=1.0, total_frames=30, codec="h264",
        )

        p1 = tmp_work_dir / "cam_dz_off.json"
        p2 = tmp_work_dir / "cam_dz_base.json"
        gen_off.generate(sample_tracks, meta, p1)
        gen_baseline.generate(sample_tracks, meta, p2)

        with open(p1) as f:
            path1 = json.load(f)
        with open(p2) as f:
            path2 = json.load(f)

        for i in range(len(path1)):
            assert abs(path1[i]["yaw"] - path2[i]["yaw"]) < 0.01
            assert abs(path1[i]["pitch"] - path2[i]["pitch"]) < 0.01


# ---------------------------------------------------------------------------
# Lookahead tests
# ---------------------------------------------------------------------------

class TestLookahead:
    """Tests for Kalman velocity lookahead."""

    def _make_gen(self, test_config, enabled=True, frames=3, max_deg=10.0):
        config = deepcopy(test_config)
        config["camera"]["lookahead_enabled"] = enabled
        config["camera"]["lookahead_frames"] = frames
        config["camera"]["lookahead_max_deg"] = max_deg
        return CameraPathGenerator(config)

    def test_leads_fast_pass(self, test_config, tmp_work_dir):
        """With lookahead, camera should lead ahead of a fast-moving ball."""
        gen_la = self._make_gen(test_config, enabled=True, frames=5)
        gen_no = self._make_gen(test_config, enabled=False)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=2.0, total_frames=60, codec="h264",
        )

        # Ball moving fast to the right
        tracks = []
        for i in range(60):
            x = 80 + i * 4  # fast rightward
            tracks.append({
                "frame": i,
                "ball": {"x": min(x, 310), "y": 80, "confidence": 0.9},
            })
        tracks_path = tmp_work_dir / "tracks_la_fast.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        p_la = tmp_work_dir / "cam_la.json"
        p_no = tmp_work_dir / "cam_no_la.json"
        gen_la.generate(tracks_path, meta, p_la)
        gen_no.generate(tracks_path, meta, p_no)

        with open(p_la) as f:
            path_la = json.load(f)
        with open(p_no) as f:
            path_no = json.load(f)

        # During steady-state fast movement, lookahead camera should be
        # ahead (higher yaw) compared to non-lookahead
        lead_frames = 0
        for i in range(15, 45):
            if path_la[i]["yaw"] > path_no[i]["yaw"] + 0.1:
                lead_frames += 1
        assert lead_frames > 15, (
            f"Lookahead camera should lead the ball; only led in {lead_frames}/30 frames"
        )

    def test_negligible_on_slow_play(self, test_config, tmp_work_dir):
        """Lookahead should have negligible effect on a stationary ball."""
        gen_la = self._make_gen(test_config, enabled=True, frames=5)
        gen_no = self._make_gen(test_config, enabled=False)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=1.0, total_frames=30, codec="h264",
        )

        # Ball stationary
        tracks = [
            {"frame": i, "ball": {"x": 200, "y": 80, "confidence": 0.9}}
            for i in range(30)
        ]
        tracks_path = tmp_work_dir / "tracks_la_slow.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        p_la = tmp_work_dir / "cam_la_slow.json"
        p_no = tmp_work_dir / "cam_no_la_slow.json"
        gen_la.generate(tracks_path, meta, p_la)
        gen_no.generate(tracks_path, meta, p_no)

        with open(p_la) as f:
            path_la = json.load(f)
        with open(p_no) as f:
            path_no = json.load(f)

        for i in range(5, 30):
            diff = abs(path_la[i]["yaw"] - path_no[i]["yaw"])
            assert diff < 1.0, (
                f"Lookahead diverged {diff:.2f}° on stationary ball at frame {i}"
            )

    def test_clamped_projection(self, test_config, tmp_work_dir):
        """Projection should be clamped to lookahead_max_deg."""
        gen = self._make_gen(test_config, enabled=True, frames=10, max_deg=2.0)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=2.0, total_frames=60, codec="h264",
        )

        # Very fast ball movement to trigger large projection
        tracks = []
        for i in range(60):
            x = 30 + i * 5
            tracks.append({
                "frame": i,
                "ball": {"x": min(x, 310), "y": 80, "confidence": 0.9},
            })
        tracks_path = tmp_work_dir / "tracks_la_clamp.json"
        with open(tracks_path, "w") as f:
            json.dump(tracks, f)

        output_path = tmp_work_dir / "cam_la_clamp.json"
        gen.generate(tracks_path, meta, output_path)

        with open(output_path) as f:
            path = json.load(f)

        # Output should still be valid and smooth (not diverge due to unclamped projection)
        for i in range(1, len(path)):
            dyaw = abs(angle_diff(path[i]["yaw"], path[i - 1]["yaw"]))
            max_delta = test_config["camera"]["max_fast_pan_speed_deg_per_sec"] / 30.0
            assert dyaw <= max_delta + 0.5, f"Yaw jump {dyaw:.2f}° at frame {i}"


# ---------------------------------------------------------------------------
# Velocity-adaptive blending tests
# ---------------------------------------------------------------------------

class TestVelocityAdaptiveBlending:
    """Tests for velocity-adaptive ball/cluster blending."""

    def _make_gen(self, test_config, enabled=True):
        config = deepcopy(test_config)
        config["center_of_play"]["enabled"] = True
        config["center_of_play"]["velocity_blend_enabled"] = enabled
        config["center_of_play"]["fast_ball_weight"] = 0.95
        config["center_of_play"]["slow_ball_weight"] = 0.50
        config["center_of_play"]["velocity_fast_thresh_deg_per_sec"] = 20.0
        config["center_of_play"]["velocity_slow_thresh_deg_per_sec"] = 2.0
        return CameraPathGenerator(config)

    def test_fast_ball_high_weight(self, test_config):
        """Fast-moving ball should result in high ball weight (low cluster influence)."""
        gen = self._make_gen(test_config, enabled=True)

        # Two frames: ball jumps far right (fast movement)
        tracks = [
            {"frame": 0, "ball": {"x": 100, "y": 80, "confidence": 0.9}},
            {"frame": 1, "ball": {"x": 200, "y": 80, "confidence": 0.9}},
        ]
        clusters = [
            {"frame": 0, "cluster": {"x": 50, "y": 80, "spread_x_deg": 20.0,
                                      "player_count": 15, "confidence": 0.6}},
            {"frame": 1, "cluster": {"x": 50, "y": 80, "spread_x_deg": 20.0,
                                      "player_count": 15, "confidence": 0.6}},
        ]

        angles = gen._tracks_to_angles_hybrid(tracks, clusters, fps=30.0)
        ball_only = gen._tracks_to_angles(tracks)

        # Frame 1 should be very close to ball-only (high ball weight)
        ball_yaw = ball_only[1][0]
        hybrid_yaw = angles[1][0]
        cluster_yaw, _ = pixel_to_yaw_pitch(50, 80, gen.det_width, gen.det_height)
        # Hybrid should be much closer to ball than cluster
        assert abs(hybrid_yaw - ball_yaw) < abs(hybrid_yaw - cluster_yaw) * 0.3

    def test_slow_ball_more_cluster(self, test_config):
        """Stationary ball should allow more cluster influence."""
        gen = self._make_gen(test_config, enabled=True)

        # Two frames: ball is completely stationary (0px movement → 0 deg/sec
        # → velocity blend uses slow_ball_weight=0.50, giving cluster 50%)
        tracks = [
            {"frame": 0, "ball": {"x": 200, "y": 80, "confidence": 0.9}},
            {"frame": 1, "ball": {"x": 200, "y": 80, "confidence": 0.9}},
        ]
        clusters = [
            {"frame": 0, "cluster": {"x": 100, "y": 80, "spread_x_deg": 20.0,
                                      "player_count": 15, "confidence": 0.6}},
            {"frame": 1, "cluster": {"x": 100, "y": 80, "spread_x_deg": 20.0,
                                      "player_count": 15, "confidence": 0.6}},
        ]

        angles_vel = gen._tracks_to_angles_hybrid(tracks, clusters, fps=30.0)

        # Compare to non-velocity (fixed blend) version
        gen_fixed = self._make_gen(test_config, enabled=False)
        angles_fixed = gen_fixed._tracks_to_angles_hybrid(tracks, clusters, fps=30.0)

        # Velocity blend at 0 speed → ball_weight=0.50, cluster=0.50
        # Fixed blend → ball_blend_weight=0.05, cluster=0.05
        # So velocity blend should pull much more toward cluster
        cluster_yaw, _ = pixel_to_yaw_pitch(100, 80, gen.det_width, gen.det_height)

        vel_dist_to_cluster = abs(angles_vel[1][0] - cluster_yaw)
        fixed_dist_to_cluster = abs(angles_fixed[1][0] - cluster_yaw)

        assert vel_dist_to_cluster < fixed_dist_to_cluster, (
            "Stationary ball should pull hybrid angle closer to cluster than fixed blend"
        )

    def test_disabled_preserves_existing(self, test_config):
        """With velocity_blend_enabled=False, output matches two-tier behavior."""
        gen_on = self._make_gen(test_config, enabled=False)  # disabled
        gen_base = CameraPathGenerator(deepcopy(test_config) | {
            "center_of_play": {
                **test_config["center_of_play"],
                "enabled": True,
            },
        })

        tracks = [
            {"frame": 0, "ball": {"x": 100, "y": 80, "confidence": 0.9}},
            {"frame": 1, "ball": {"x": 200, "y": 80, "confidence": 0.9}},
        ]
        clusters = [
            {"frame": 0, "cluster": {"x": 50, "y": 80, "spread_x_deg": 20.0,
                                      "player_count": 15, "confidence": 0.6}},
            {"frame": 1, "cluster": {"x": 50, "y": 80, "spread_x_deg": 20.0,
                                      "player_count": 15, "confidence": 0.6}},
        ]

        angles_off = gen_on._tracks_to_angles_hybrid(tracks, clusters, fps=30.0)
        angles_base = gen_base._tracks_to_angles_hybrid(tracks, clusters, fps=30.0)

        for i in range(len(angles_off)):
            assert abs(angles_off[i][0] - angles_base[i][0]) < 0.01
            assert abs(angles_off[i][1] - angles_base[i][1]) < 0.01
