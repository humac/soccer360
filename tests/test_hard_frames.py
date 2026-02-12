"""Tests for hard frame identification and export."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.hard_frames import HardFrameExporter
from src.utils import VideoMeta


@pytest.fixture
def meta():
    return VideoMeta(
        width=320, height=160, fps=10.0,
        duration=3.0, total_frames=30, codec="h264",
    )


@pytest.fixture
def exporter(test_config, tmp_path):
    config = {
        **test_config,
        "paths": {**test_config["paths"], "labeling": str(tmp_path / "labeling")},
    }
    return HardFrameExporter(config)


class TestHardFrameIdentification:
    """Test the _identify_hard_frames logic without extracting actual frames."""

    def test_low_confidence_detected(self, exporter, meta):
        """Frames with confidence below threshold should be flagged."""
        detections = [
            {"frame": 0, "bbox": [10, 10, 20, 20], "confidence": 0.9, "class": 0},
            {"frame": 1, "bbox": [10, 10, 20, 20], "confidence": 0.1, "class": 0},
            {"frame": 2, "bbox": [10, 10, 20, 20], "confidence": 0.2, "class": 0},
            {"frame": 3, "bbox": [10, 10, 20, 20], "confidence": 0.9, "class": 0},
        ]
        tracks = [
            {"frame": i, "ball": {"x": 15, "y": 15, "bbox": [10, 10, 20, 20],
                                   "confidence": d["confidence"], "track_id": 1}}
            for i, d in enumerate(detections)
        ]

        hard = exporter._identify_hard_frames(detections, tracks, meta)
        hard_indices = {h["frame_index"] for h in hard}

        assert 1 in hard_indices  # conf 0.1 < 0.3
        assert 2 in hard_indices  # conf 0.2 < 0.3
        assert 0 not in hard_indices  # conf 0.9 >= 0.3

        # Check reason
        for h in hard:
            if h["frame_index"] in (1, 2):
                assert "low_confidence" in h["reasons"]

    def test_gap_frames_detected(self, exporter, meta):
        """Consecutive lost-ball gaps exceeding threshold should be flagged."""
        # gap_frames=5 in test_config
        tracks = []
        for i in range(20):
            if 5 <= i < 12:  # 7 consecutive lost frames (> gap_frames=5)
                tracks.append({"frame": i, "ball": None})
            else:
                tracks.append({
                    "frame": i,
                    "ball": {"x": 15, "y": 15, "bbox": [10, 10, 20, 20],
                             "confidence": 0.9, "track_id": 1},
                })

        hard = exporter._identify_hard_frames([], tracks, meta)
        hard_indices = {h["frame_index"] for h in hard}

        # All frames in the gap should be flagged
        for f in range(5, 12):
            assert f in hard_indices
        # Frames outside the gap should NOT be flagged
        assert 4 not in hard_indices
        assert 12 not in hard_indices

        # Check reason
        for h in hard:
            if 5 <= h["frame_index"] < 12:
                assert "lost_ball_gap" in h["reasons"]

    def test_short_gap_not_flagged(self, exporter, meta):
        """Gaps shorter than threshold should not be flagged."""
        # gap_frames=5 in test_config, gap of 3 should not trigger
        tracks = []
        for i in range(10):
            if 3 <= i < 6:  # 3 consecutive lost frames (< gap_frames=5)
                tracks.append({"frame": i, "ball": None})
            else:
                tracks.append({
                    "frame": i,
                    "ball": {"x": 15, "y": 15, "bbox": [10, 10, 20, 20],
                             "confidence": 0.9, "track_id": 1},
                })

        hard = exporter._identify_hard_frames([], tracks, meta)
        gap_frames = [h for h in hard if "lost_ball_gap" in h.get("reasons", [])]
        assert len(gap_frames) == 0

    def test_position_jumps_detected(self, exporter, meta):
        """Large position jumps between consecutive frames should be flagged."""
        tracks = [
            {"frame": 0, "ball": {"x": 10, "y": 10, "bbox": [5, 5, 15, 15],
                                   "confidence": 0.9, "track_id": 1}},
            {"frame": 1, "ball": {"x": 15, "y": 10, "bbox": [10, 5, 20, 15],
                                   "confidence": 0.9, "track_id": 1}},
            # Big jump: from (15,10) to (200,10) = 185px > 150px threshold
            {"frame": 2, "ball": {"x": 200, "y": 10, "bbox": [195, 5, 205, 15],
                                   "confidence": 0.9, "track_id": 1}},
            {"frame": 3, "ball": {"x": 205, "y": 10, "bbox": [200, 5, 210, 15],
                                   "confidence": 0.9, "track_id": 1}},
        ]

        hard = exporter._identify_hard_frames([], tracks, meta)
        hard_indices = {h["frame_index"] for h in hard}

        assert 2 in hard_indices  # Jump frame
        assert 0 not in hard_indices
        assert 1 not in hard_indices

        for h in hard:
            if h["frame_index"] == 2:
                assert "position_jump" in h["reasons"]
                assert h["jump_distance_px"] > 150

    def test_max_export_sampling(self, test_config, tmp_path, meta):
        """When hard frames exceed max_export_frames, sampling should reduce count."""
        config = {
            **test_config,
            "paths": {**test_config["paths"], "labeling": str(tmp_path / "labeling")},
            "active_learning": {
                **test_config["active_learning"],
                "max_export_frames": 5,
                "gap_frames": 1,
            },
        }
        exp = HardFrameExporter(config)

        # Create 20 lost frames -> all should be hard frames
        tracks = [{"frame": i, "ball": None} for i in range(20)]

        hard = exp._identify_hard_frames([], tracks, meta)
        assert len(hard) == 20  # All identified

        # The run() method does the sampling, not _identify_hard_frames
        # We test that run handles sampling by mocking extract_frame
        detections_path = tmp_path / "detections.jsonl"
        detections_path.write_text("")
        tracks_path = tmp_path / "tracks.json"
        tracks_path.write_text(json.dumps(tracks))
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        with patch("src.hard_frames.extract_frame"):
            exp.run(str(tmp_path / "test.mp4"), meta, detections_path, tracks_path, work_dir)

        manifest = json.loads(
            (tmp_path / "labeling" / "test" / "hard_frames.json").read_text()
        )
        assert manifest["hard_frames_exported"] <= 5

    def test_manifest_structure(self, test_config, tmp_path, meta):
        """hard_frames.json manifest should have required fields."""
        config = {
            **test_config,
            "paths": {**test_config["paths"], "labeling": str(tmp_path / "labeling")},
        }
        exp = HardFrameExporter(config)

        detections = [
            {"frame": 0, "bbox": [10, 10, 20, 20], "confidence": 0.1, "class": 0},
        ]
        tracks = [
            {"frame": 0, "ball": {"x": 15, "y": 15, "bbox": [10, 10, 20, 20],
                                   "confidence": 0.1, "track_id": 1}},
        ]

        detections_path = tmp_path / "detections.jsonl"
        detections_path.write_text(json.dumps(detections[0]) + "\n")
        tracks_path = tmp_path / "tracks.json"
        tracks_path.write_text(json.dumps(tracks))
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        with patch("src.hard_frames.extract_frame"):
            exp.run(str(tmp_path / "test.mp4"), meta, detections_path, tracks_path, work_dir)

        manifest_path = tmp_path / "labeling" / "test" / "hard_frames.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())

        assert "video" in manifest
        assert "match_name" in manifest
        assert "total_frames_analyzed" in manifest
        assert "hard_frames_identified" in manifest
        assert "hard_frames_exported" in manifest
        assert "criteria" in manifest
        assert "frames" in manifest

        # Check frame entry structure
        for frame in manifest["frames"]:
            assert "frame_index" in frame
            assert "timestamp_sec" in frame
            assert "reasons" in frame

    def test_disabled_skips_export(self, test_config, tmp_path, meta):
        """When active_learning.enabled=False, no frames should be exported."""
        config = {
            **test_config,
            "paths": {**test_config["paths"], "labeling": str(tmp_path / "labeling")},
            "active_learning": {**test_config["active_learning"], "enabled": False},
        }
        exp = HardFrameExporter(config)

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        detections_path = tmp_path / "detections.jsonl"
        detections_path.write_text("")
        tracks_path = tmp_path / "tracks.json"
        tracks_path.write_text("[]")

        exp.run(str(tmp_path / "test.mp4"), meta, detections_path, tracks_path, work_dir)

        labeling_dir = tmp_path / "labeling" / "test"
        assert not labeling_dir.exists()
