"""Shared test fixtures: synthetic test video, config, detection data."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="session")
def test_config() -> dict:
    """Minimal pipeline config for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "paths": {
                "ingest": f"{tmpdir}/ingest",
                "scratch": f"{tmpdir}/scratch",
                "processed": f"{tmpdir}/processed",
                "highlights": f"{tmpdir}/highlights",
                "models": f"{tmpdir}/models",
                "labeling": f"{tmpdir}/labeling",
                "stagging": f"{tmpdir}/stagging",
                "archive_raw": f"{tmpdir}/archive_raw",
                "logs": None,
            },
            "model": {
                "path": "yolov8s.pt",
                "base_model": "yolov8s.pt",
                "use_tensorrt": False,
            },
            "detector": {
                "runtime_override_path": f"{tmpdir}/data/ingest_model_selection.json",
                "player_runtime_override_path": f"{tmpdir}/data/ingest_player_model_selection.json",
                "ball_model_config_override_path": f"{tmpdir}/data/ball_model_config.json",
                "batch_size": 4,
                "resolution": [320, 160],
                "confidence_threshold": 0.25,
                "nms_iou_threshold": 0.45,
                "process_every_n_frames": 1,
                "tiling": {
                    "enabled": False,
                    "grid": [2, 2],
                    "overlap": 0.1,
                    "equirect_aware_overlap": False,
                    "edge_overlap_boost": 1.5,
                },
            },
            "field_of_interest": {
                "enabled": False,
                "center_mode": "fixed",
                "center_yaw_deg": 0,
                "yaw_window_deg": 200,
                "pitch_min_deg": -45,
                "pitch_max_deg": 20,
                "auto_sample_seconds": 30,
                "auto_min_conf": 0.25,
            },
            "tracker": {
                "track_high_thresh": 0.25,
                "track_low_thresh": 0.1,
                "new_track_thresh": 0.25,
                "track_buffer": 30,
                "match_thresh": 0.4,
                "max_speed_px_per_frame": 200,
                "max_displacement_px": 300,
                "min_bbox_area": 10,
                "max_bbox_area": 10000,
            },
            "camera": {
                "max_pan_speed_deg_per_sec": 45.0,
                "max_fast_pan_speed_deg_per_sec": 90.0,
                "ema_alpha": 0.10,
                "default_fov": 90.0,
                "min_fov": 80.0,
                "max_fov": 100.0,
                "lost_coast_frames": 30,
                "lost_drift_frames": 90,
                "field_center_yaw_deg": 0.0,
                "field_center_pitch_deg": -5.0,
                "deadband_deg": 2.5,
                "velocity_threshold_deg_per_sec": 4.0,
                "fov_ema_alpha": 0.08,
                "spatial_deadzone_enabled": False,
                "spatial_deadzone_frac": 0.30,
                "spatial_deadzone_ramp": 0.20,
                "lookahead_enabled": False,
                "lookahead_frames": 3,
                "lookahead_max_deg": 10.0,
                "kalman": {
                    "process_noise": 0.1,
                    "measurement_noise": 2.0,
                },
            },
            "reframer": {
                "output_resolution": [320, 180],
                "source_downscale": [640, 320],
                "num_workers": 2,
                "interpolation": "bilinear",
                "tactical_fov": 120,
                "tactical_yaw": 0.0,
                "tactical_pitch": -5.0,
            },
            "highlights": {
                "speed_percentile": 95,
                "direction_change_deg": 90,
                "goal_box_regions": [
                    [0.0, 0.3, 0.08, 0.7],
                    [0.92, 0.3, 1.0, 0.7],
                ],
                "pre_margin_sec": 1.0,
                "post_margin_sec": 0.5,
                "min_clip_gap_sec": 2.0,
                "min_clip_duration_sec": 1.0,
                "cluster_convergence_window": 5,
                "cluster_convergence_deg": 5.0,
                "cluster_velocity_window": 3,
                "cluster_velocity_deg_per_sec": 10.0,
                "cluster_goal_zone_regions": None,
                "cluster_density_percentile": 80,
                "camera_motion_window": 3,
                "camera_motion_deg_per_sec": 8.0,
                "camera_zoom_delta": 2.0,
                "same_type_cooldown_sec": 0.75,
                "motion_only_penalty": 0.8,
                "score_weights": {
                    "speed": 1.0,
                    "goal_box": 1.5,
                    "direction_change": 0.8,
                    "cluster_convergence": 1.2,
                    "cluster_velocity": 0.7,
                    "cluster_goal_zone": 1.3,
                    "cluster_density": 0.5,
                    "camera_motion": 0.8,
                },
                "combined_signal_bonus": 1.5,
                "min_clip_score": 0.5,
                "max_clips": 10,
            },
            "exporter": {
                "codec": "libx264",
                "crf": 23,
                "preset": "ultrafast",
                "archive_raw": False,
                "delete_raw": False,
            },
            "watcher": {
                "extensions": [".mp4"],
                "stability_checks": 2,
                "stability_interval_sec": 0.5,
                "processed_state_file": "watcher_processed_ingest.json",
                "processed_state_max_entries": 50000,
            },
            "ingest": {
                "archive_on_success": False,
                "archive_dir": f"{tmpdir}/archive_raw",
                "archive_mode": "leave",
                "archive_name_template": "{match}_{job_id}{ext}",
                "archive_collision": "suffix",
            },
            "active_learning": {
                "enabled": True,
                # Legacy keys (for HardFrameExporter tests)
                "confidence_threshold": 0.3,
                "gap_frames": 5,
                "max_export_frames": 50,
                "position_jump_px": 150,
                # V1 keys (for ActiveLearningExporter)
                "export_dir": f"{tmpdir}/labeling",
                "export_max_frames": 50,
                "export_every_n_frames": 2,
                "low_conf_min": 0.20,
                "low_conf_max": 0.50,
                "lost_run_frames": 5,
                "jump_trigger_px": 200,
            },
            "detection": {
                "path": "yolov8s.pt",
                "classes": [32, 0],
                "conf": 0.35,
                "iou": 0.5,
                "img_size": 160,
                "max_det": 50,
                "half": False,
                "device": "cpu",
                "ball_model": {
                    "type": "yolo",
                    "path": None,
                    "input_height": 288,
                    "input_width": 512,
                    "buffer_size": 3,
                    "heatmap_threshold": 0.5,
                    "peak_radius": 5,
                    "synthetic_bbox_half": 5,
                },
            },
            "filters": {
                "min_y_frac": 0.20,
                "max_y_frac": 0.85,
                "max_jump_px": 250,
                "max_speed_px_per_s": 2500,
                "jump_max_gap_frames": 15,
            },
            "tracking": {
                "ema_alpha": 0.35,
                "require_persistence": 2,
                "window": 3,
            },
            "center_of_play": {
                "enabled": True,
                "player_class": 0,
                "min_player_conf": 0.60,
                "trim_fraction": 0.25,
                "min_players": 5,
                "ball_blend_weight": 0.05,
                "low_conf_ball_blend_weight": 0.20,
                "ema_alpha": 0.15,
                "fov_from_spread": True,
                "spread_max_fov": 105.0,
                "spread_min_deg": 15.0,
                "spread_max_deg": 60.0,
                "velocity_blend_enabled": False,
                "fast_ball_weight": 0.95,
                "slow_ball_weight": 0.50,
                "velocity_fast_thresh_deg_per_sec": 20.0,
                "velocity_slow_thresh_deg_per_sec": 2.0,
            },
            "mode": {
                "allow_no_model": True,
            },
            "logging": {
                "level": "WARNING",
                "file": None,
            },
        }
        yield config


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    """Create a temporary working directory."""
    work = tmp_path / "work"
    work.mkdir()
    return work


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """Create a short synthetic equirectangular video with a moving circle (ball).

    3 seconds at 10 fps, 640x320 resolution (small for fast tests).
    A white circle moves across the frame from left to right.
    """
    np = pytest.importorskip("numpy")
    tmpdir = tmp_path_factory.mktemp("video")
    output = tmpdir / "test_equirect.mp4"

    fps = 10
    duration = 3
    w, h = 640, 320
    total_frames = fps * duration

    # Create frames and pipe to ffmpeg
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(output),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in range(total_frames):
        # Green field background
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = [34, 139, 34]  # Forest green

        # White circle moving left to right
        cx = int((i / total_frames) * w * 0.8 + w * 0.1)
        cy = h // 2
        radius = 8

        # Draw circle (simple rasterization)
        yy, xx = np.ogrid[-cy:h - cy, -cx:w - cx]
        mask = xx * xx + yy * yy <= radius * radius
        frame[mask] = [255, 255, 255]

        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    return output


@pytest.fixture
def sample_detections(tmp_work_dir: Path) -> Path:
    """Create sample detection JSONL file."""
    detections = []
    for frame in range(30):
        # Ball moving from left to right across 320x160 detection space
        x = 20 + frame * 9  # ~20 to ~280
        y = 80  # center vertically
        detections.append({
            "frame": frame,
            "bbox": [x - 5, y - 5, x + 5, y + 5],
            "confidence": 0.85 if frame != 15 else 0.15,  # one low-conf frame
            "class": 32,
        })

    # Frame 10 has no detection (ball lost)
    detections = [d for d in detections if d["frame"] != 10]

    path = tmp_work_dir / "detections.jsonl"
    with open(path, "w") as f:
        for d in detections:
            f.write(json.dumps(d) + "\n")
    return path


@pytest.fixture
def sample_tracks(tmp_work_dir: Path) -> Path:
    """Create sample tracks JSON file."""
    tracks = []
    for frame in range(30):
        if frame == 10:
            tracks.append({"frame": frame, "ball": None})
        else:
            x = 20 + frame * 9
            y = 80
            tracks.append({
                "frame": frame,
                "ball": {
                    "x": x,
                    "y": y,
                    "bbox": [x - 5, y - 5, x + 5, y + 5],
                    "confidence": 0.85,
                    "track_id": 1,
                },
            })

    path = tmp_work_dir / "tracks.json"
    with open(path, "w") as f:
        json.dump(tracks, f)
    return path


@pytest.fixture
def sample_cluster_data(tmp_work_dir: Path) -> Path:
    """Create sample player cluster JSON with convergence and goal-zone events.

    30 frames at 320x160 detection space:
    - Frames 0-11: stable spread ~35 deg, centroid at mid-field
    - Frames 12-18: convergence event (spread drops 40->10 deg)
    - Frames 19-24: stable again, centroid drifting right
    - Frames 25-29: centroid near right goal zone, high player count
    """
    clusters = []
    for frame in range(30):
        # Default: mid-field centroid with moderate spread
        x = 160.0
        y = 80.0
        spread = 35.0
        count = 10
        conf = 0.7

        if 12 <= frame <= 18:
            # Convergence event: spread drops rapidly
            spread = 40.0 - (frame - 12) * 5.0  # 40 -> 10
            count = 12 + (frame - 12)  # increasing density
        elif frame >= 25:
            # Near right goal zone (x > 92% of 320 = 294.4)
            x = 300.0
            count = 15  # high player count
            spread = 20.0

        clusters.append({
            "frame": frame,
            "cluster": {
                "x": x,
                "y": y,
                "spread_x_deg": spread,
                "player_count": count,
                "confidence": conf,
            },
        })

    path = tmp_work_dir / "player_cluster.json"
    with open(path, "w") as f:
        json.dump(clusters, f)
    return path
