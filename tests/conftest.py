"""Shared test fixtures: synthetic test video, config, detection data."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
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
                "archive_raw": f"{tmpdir}/archive_raw",
                "logs": None,
            },
            "model": {
                "path": "yolov8s.pt",
                "base_model": "yolov8s.pt",
                "use_tensorrt": False,
            },
            "detector": {
                "batch_size": 4,
                "resolution": [320, 160],
                "confidence_threshold": 0.25,
                "nms_iou_threshold": 0.45,
                "tiling": {"enabled": False},
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
                "max_pan_speed_deg_per_sec": 60.0,
                "max_fast_pan_speed_deg_per_sec": 120.0,
                "ema_alpha": 0.15,
                "default_fov": 90.0,
                "min_fov": 80.0,
                "max_fov": 100.0,
                "lost_coast_frames": 30,
                "lost_drift_frames": 90,
                "field_center_yaw_deg": 0.0,
                "field_center_pitch_deg": -5.0,
                "kalman": {
                    "process_noise": 0.1,
                    "measurement_noise": 1.0,
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
            "class": 0,
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
