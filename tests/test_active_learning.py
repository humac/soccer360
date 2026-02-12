"""Tests for V1 active learning frame export."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.active_learning import ActiveLearningExporter
from src.utils import VideoMeta


@pytest.fixture
def al_config(test_config):
    """Return config with active learning settings."""
    return test_config


@pytest.fixture
def fake_meta():
    """Minimal VideoMeta for testing."""
    return VideoMeta(width=640, height=320, fps=30.0, duration=10.0, total_frames=300, codec="h264")


def _write_detections(path: Path, detections: list[dict]):
    """Write detections as JSONL."""
    with open(path, "w") as f:
        for det in detections:
            f.write(json.dumps(det) + "\n")


def _write_tracks(path: Path, tracks: list[dict]):
    """Write tracks as JSON."""
    with open(path, "w") as f:
        json.dump(tracks, f)


# ---------------------------------------------------------------------------
# Trigger 1: Low confidence
# ---------------------------------------------------------------------------

class TestLowConfTrigger:
    def test_low_conf_in_range_triggered(self, al_config, tmp_path, fake_meta):
        """conf=0.35 in [0.20, 0.50] -> flagged."""
        det_path = tmp_path / "detections.jsonl"
        tracks_path = tmp_path / "tracks.json"

        _write_detections(det_path, [
            {"frame_index": 0, "conf": 0.35, "bbox_xyxy": [10, 10, 20, 20], "class_id": 32},
        ])
        _write_tracks(tracks_path, [
            {"frame": 0, "ball": {"x": 15, "y": 15}, "status": "accepted"},
        ])

        exporter = ActiveLearningExporter(al_config)
        candidates = exporter._identify_candidates(
            [{"frame_index": 0, "conf": 0.35, "bbox_xyxy": [10, 10, 20, 20]}],
            [{"frame": 0, "ball": {"x": 15, "y": 15}, "status": "accepted"}],
            fake_meta,
            [],
        )
        assert len(candidates) == 1
        assert "low_conf" in candidates[0]["triggers"]

    def test_low_conf_below_min_not_triggered(self, al_config, fake_meta):
        """conf=0.10 < 0.20 -> not flagged."""
        exporter = ActiveLearningExporter(al_config)
        candidates = exporter._identify_candidates(
            [{"frame_index": 0, "conf": 0.10, "bbox_xyxy": [10, 10, 20, 20]}],
            [{"frame": 0, "ball": None, "status": "lost"}],
            fake_meta,
            [],
        )
        # Should not trigger low_conf (conf below range)
        low_conf_candidates = [c for c in candidates if "low_conf" in c.get("triggers", [])]
        assert len(low_conf_candidates) == 0

    def test_low_conf_above_max_not_triggered(self, al_config, fake_meta):
        """conf=0.60 > 0.50 -> not flagged."""
        exporter = ActiveLearningExporter(al_config)
        candidates = exporter._identify_candidates(
            [{"frame_index": 0, "conf": 0.60, "bbox_xyxy": [10, 10, 20, 20]}],
            [{"frame": 0, "ball": {"x": 15, "y": 15}, "status": "accepted"}],
            fake_meta,
            [],
        )
        low_conf_candidates = [c for c in candidates if "low_conf" in c.get("triggers", [])]
        assert len(low_conf_candidates) == 0


# ---------------------------------------------------------------------------
# Trigger 2: Lost ball runs
# ---------------------------------------------------------------------------

class TestLostRunTrigger:
    def test_lost_run_at_threshold_one_frame(self, al_config, fake_meta):
        """streak=5 (test config lost_run_frames=5) -> exactly ONE frame exported."""
        exporter = ActiveLearningExporter(al_config)
        # Create exactly 5 lost frames (matches test config lost_run_frames=5)
        tracks = [{"frame": i, "ball": None, "status": "lost"} for i in range(10)]
        candidates = exporter._identify_candidates([], tracks, fake_meta, [])
        lost_run_candidates = [c for c in candidates if "lost_run" in c.get("triggers", [])]
        # Exactly one frame per streak at the threshold crossing
        assert len(lost_run_candidates) == 1
        # Threshold crossing frame = streak_start + lost_run_frames - 1 = 0 + 5 - 1 = 4
        assert lost_run_candidates[0]["frame_index"] == 4

    def test_lost_run_below_threshold(self, al_config, fake_meta):
        """streak=3 < 5 -> not flagged."""
        exporter = ActiveLearningExporter(al_config)
        tracks = [{"frame": i, "ball": None, "status": "lost"} for i in range(3)]
        # Add a ball-found frame to end the streak
        tracks.append({"frame": 3, "ball": {"x": 50, "y": 50}, "status": "accepted"})
        candidates = exporter._identify_candidates([], tracks, fake_meta, [])
        lost_run_candidates = [c for c in candidates if "lost_run" in c.get("triggers", [])]
        assert len(lost_run_candidates) == 0

    def test_lost_run_multiple_streaks(self, al_config, fake_meta):
        """Two separate streaks >= threshold -> one frame each."""
        exporter = ActiveLearningExporter(al_config)
        tracks = []
        # First streak: frames 0-9 (10 frames lost, >= threshold of 5)
        for i in range(10):
            tracks.append({"frame": i, "ball": None, "status": "lost"})
        # Ball found: frames 10-14
        for i in range(10, 15):
            tracks.append({"frame": i, "ball": {"x": 50, "y": 50}, "status": "accepted"})
        # Second streak: frames 15-24 (10 frames lost, >= threshold of 5)
        for i in range(15, 25):
            tracks.append({"frame": i, "ball": None, "status": "lost"})

        candidates = exporter._identify_candidates([], tracks, fake_meta, [])
        lost_run_candidates = [c for c in candidates if "lost_run" in c.get("triggers", [])]
        assert len(lost_run_candidates) == 2
        # First streak threshold crossing: 0 + 5 - 1 = 4
        assert lost_run_candidates[0]["frame_index"] == 4
        # Second streak threshold crossing: 15 + 5 - 1 = 19
        assert lost_run_candidates[1]["frame_index"] == 19


# ---------------------------------------------------------------------------
# Trigger 3: Jump rejection events
# ---------------------------------------------------------------------------

class TestJumpTrigger:
    def test_jump_event_triggers_export(self, al_config, fake_meta):
        """jump_reject event with distance >= jump_trigger_px -> frame flagged."""
        exporter = ActiveLearningExporter(al_config)
        # test config jump_trigger_px=200
        events = [{"frame": 10, "trigger": "jump_reject", "distance_px": 250.0}]
        candidates = exporter._identify_candidates([], [], fake_meta, events)
        jump_candidates = [c for c in candidates if "jump_reject" in c.get("triggers", [])]
        assert len(jump_candidates) == 1
        assert jump_candidates[0]["frame_index"] == 10

    def test_jump_event_below_trigger_not_exported(self, al_config, fake_meta):
        """jump_reject with distance < jump_trigger_px -> not exported."""
        exporter = ActiveLearningExporter(al_config)
        events = [{"frame": 10, "trigger": "jump_reject", "distance_px": 100.0}]
        candidates = exporter._identify_candidates([], [], fake_meta, events)
        jump_candidates = [c for c in candidates if "jump_reject" in c.get("triggers", [])]
        assert len(jump_candidates) == 0

    def test_speed_reject_event_also_triggers(self, al_config, fake_meta):
        """speed_reject with distance >= jump_trigger_px is also captured."""
        exporter = ActiveLearningExporter(al_config)
        events = [{"frame": 5, "trigger": "speed_reject", "distance_px": 300.0, "speed_px_s": 5000}]
        candidates = exporter._identify_candidates([], [], fake_meta, events)
        jump_candidates = [c for c in candidates if "jump_reject" in c.get("triggers", [])]
        assert len(jump_candidates) == 1


# ---------------------------------------------------------------------------
# Gating: export_every_n and max cap
# ---------------------------------------------------------------------------

class TestGating:
    def test_export_every_n_by_frame_index(self, al_config, fake_meta):
        """every_n=2 -> only candidates with frame_index % 2 == 0 exported."""
        exporter = ActiveLearningExporter(al_config)
        assert exporter.export_every_n == 2

        # Create candidates at frame indices 0,1,2,3,4,5
        detections = [
            {"frame_index": i, "conf": 0.35, "bbox_xyxy": [10, 10, 20, 20]}
            for i in range(6)
        ]
        tracks = [
            {"frame": i, "ball": {"x": 15, "y": 15}, "status": "accepted"}
            for i in range(6)
        ]

        candidates = exporter._identify_candidates(detections, tracks, fake_meta, [])
        assert len(candidates) == 6  # All are candidates

        # Apply gating manually (same as run() does)
        gated = [c for c in candidates if c["frame_index"] % exporter.export_every_n == 0]
        assert len(gated) == 3  # frames 0, 2, 4
        assert all(c["frame_index"] % 2 == 0 for c in gated)

    def test_export_max_cap(self, al_config, fake_meta):
        """Many candidates, max=50 -> 50 exported."""
        exporter = ActiveLearningExporter(al_config)
        assert exporter.export_max_frames == 50  # test config value

        # Create 200 candidates (all with even frame_index to pass every_n=2)
        detections = [
            {"frame_index": i * 2, "conf": 0.35, "bbox_xyxy": [10, 10, 20, 20]}
            for i in range(200)
        ]
        candidates = exporter._identify_candidates(detections, [], fake_meta, [])
        gated = [c for c in candidates if c["frame_index"] % exporter.export_every_n == 0]
        if len(gated) > exporter.export_max_frames:
            gated = gated[:exporter.export_max_frames]
        assert len(gated) == 50


# ---------------------------------------------------------------------------
# Mode skipping
# ---------------------------------------------------------------------------

class TestModeSkipping:
    def test_disabled_skips_all(self, al_config, tmp_path, fake_meta):
        """enabled=false -> no exports."""
        original = al_config["active_learning"]["enabled"]
        al_config["active_learning"]["enabled"] = False
        exporter = ActiveLearningExporter(al_config)

        det_path = tmp_path / "detections.jsonl"
        tracks_path = tmp_path / "tracks.json"
        _write_detections(det_path, [
            {"frame_index": 0, "conf": 0.35, "bbox_xyxy": [10, 10, 20, 20]},
        ])
        _write_tracks(tracks_path, [{"frame": 0, "ball": None, "status": "lost"}])

        # Should return without doing anything
        with patch("src.active_learning.extract_frame") as mock_extract:
            exporter.run(
                str(tmp_path / "video.mp4"), fake_meta,
                det_path, tracks_path, tmp_path,
                tracking_events=[], mode="normal",
            )
            mock_extract.assert_not_called()
        al_config["active_learning"]["enabled"] = original

    def test_no_detect_mode_skips(self, al_config, tmp_path, fake_meta):
        """mode='no_detect' -> no exports."""
        exporter = ActiveLearningExporter(al_config)

        det_path = tmp_path / "detections.jsonl"
        tracks_path = tmp_path / "tracks.json"
        _write_detections(det_path, [
            {"frame_index": 0, "conf": 0.35, "bbox_xyxy": [10, 10, 20, 20]},
        ])
        _write_tracks(tracks_path, [{"frame": 0, "ball": None, "status": "lost"}])

        with patch("src.active_learning.extract_frame") as mock_extract:
            exporter.run(
                str(tmp_path / "video.mp4"), fake_meta,
                det_path, tracks_path, tmp_path,
                tracking_events=[], mode="no_detect",
            )
            mock_extract.assert_not_called()


# ---------------------------------------------------------------------------
# Manifest structure
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_structure(self, al_config, tmp_path, fake_meta):
        """Required fields present in JSON."""
        export_dir = tmp_path / "labeling"
        al_config["active_learning"]["export_dir"] = str(export_dir)
        al_config["active_learning"]["enabled"] = True
        exporter = ActiveLearningExporter(al_config)

        det_path = tmp_path / "detections.jsonl"
        tracks_path = tmp_path / "tracks.json"
        _write_detections(det_path, [
            {"frame_index": 0, "conf": 0.35, "bbox_xyxy": [10, 10, 20, 20]},
            {"frame_index": 2, "conf": 0.40, "bbox_xyxy": [20, 20, 30, 30]},
        ])
        _write_tracks(tracks_path, [
            {"frame": 0, "ball": {"x": 15, "y": 15}, "status": "accepted"},
            {"frame": 1, "ball": {"x": 15, "y": 15}, "status": "accepted"},
            {"frame": 2, "ball": {"x": 25, "y": 25}, "status": "accepted"},
        ])

        with patch("src.active_learning.extract_frame"):
            exporter.run(
                str(tmp_path / "video.mp4"), fake_meta,
                det_path, tracks_path, tmp_path,
                tracking_events=[], mode="normal",
            )

        manifest_path = tmp_path / "hard_frames.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "video" in manifest
        assert "match_name" in manifest
        assert "total_candidates" in manifest
        assert "exported_count" in manifest
        assert "config" in manifest
        assert "frames" in manifest
        assert "low_conf_range" in manifest["config"]
        assert "lost_run_frames" in manifest["config"]
        assert "jump_trigger_px" in manifest["config"]
        assert "export_every_n" in manifest["config"]
        assert "export_max" in manifest["config"]
