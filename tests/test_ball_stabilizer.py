"""Tests for V1 BallStabilizer: persistence gate, jump/speed rejection, EMA."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tracker import BallStabilizer


def _write_v1_detections(path: Path, detections: list[dict]):
    """Write V1-format detections JSONL."""
    with open(path, "w") as f:
        for d in detections:
            f.write(json.dumps(d) + "\n")


def _make_det(frame: int, cx: float, cy: float, conf: float = 0.7) -> dict:
    """Create a V1-format detection."""
    return {
        "frame_index": frame,
        "time_sec": frame / 10.0,
        "bbox_xyxy": [cx - 5, cy - 5, cx + 5, cy + 5],
        "conf": conf,
        "class_id": 32,
    }


class TestPersistenceGate:
    def test_rejects_single(self, tmp_path):
        """1 det in window=3 with require=2 → rejected with reason=persistence."""
        config = {
            "tracking": {"ema_alpha": 0.35, "require_persistence": 2, "window": 3},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        # Only frame 0 has detection, frames 1-2 have none
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, [_make_det(0, 50, 50)])

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        assert len(tracks) == 1
        assert tracks[0]["status"] == "rejected"
        assert tracks[0]["reason"] == "persistence"
        assert tracks[0]["ball"] is None
        assert tracks[0]["raw_det"] is not None

    def test_accepts_when_persistent(self, tmp_path):
        """2+ dets in window=3 → accepted."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 2, "window": 3},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        dets = [_make_det(0, 50, 50), _make_det(1, 52, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        # Frame 0: 1 det in window → rejected (persistence)
        assert tracks[0]["status"] == "rejected"
        assert tracks[0]["reason"] == "persistence"
        # Frame 1: 2 dets in window → accepted
        assert tracks[1]["status"] == "accepted"
        assert tracks[1]["ball"] is not None

    def test_window_slides(self, tmp_path):
        """Old detections fall out of window."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 2, "window": 3},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        # Dets at frames 0, 1 (accepted at frame 1), then gap 2-4, det at 5
        dets = [_make_det(0, 50, 50), _make_det(1, 52, 50), _make_det(5, 60, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        # Frame 5: window=[False, False, True] → only 1 det, require 2 → rejected
        assert tracks[5]["status"] == "rejected"
        assert tracks[5]["reason"] == "persistence"


class TestJumpRejection:
    def test_large_jump_rejected(self, tmp_path):
        """Distance > max_jump_px → rejected with reason=jump + event."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 50, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        # Frame 0 at (50,50), frame 1 at (200,50) → distance=150 > 50
        dets = [_make_det(0, 50, 50), _make_det(1, 200, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        events = stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[0]["status"] == "accepted"
        assert tracks[1]["status"] == "rejected"
        assert tracks[1]["reason"] == "jump"

        # Event emitted
        assert len(events) >= 1
        assert events[0]["trigger"] == "jump_reject"
        assert events[0]["distance_px"] == 150.0

    def test_speed_rejected(self, tmp_path):
        """Speed > max_speed_px_per_s → rejected with reason=speed."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 100},
        }
        stab = BallStabilizer(config)

        # Frame 0→1 at 10fps: move 50px in 0.1s = 500 px/s > 100
        dets = [_make_det(0, 50, 50), _make_det(1, 100, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        events = stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[1]["status"] == "rejected"
        assert tracks[1]["reason"] == "speed"
        assert any(e["trigger"] == "speed_reject" for e in events)

    def test_normal_motion_accepted(self, tmp_path):
        """Small move → accepted."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 250, "max_speed_px_per_s": 2500},
        }
        stab = BallStabilizer(config)

        # Small move: 5px at 10fps = 50 px/s
        dets = [_make_det(0, 50, 50), _make_det(1, 55, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[0]["status"] == "accepted"
        assert tracks[1]["status"] == "accepted"


class TestFirstDetection:
    def test_rejected_when_persistence_gt1(self, tmp_path):
        """require_persistence=2 → first detection rejected (persistence)."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 2, "window": 3},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        dets = [_make_det(0, 50, 50), _make_det(1, 52, 50), _make_det(2, 54, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[0]["status"] == "rejected"
        assert tracks[0]["reason"] == "persistence"

    def test_accepted_when_persistence_eq1(self, tmp_path):
        """require_persistence=1 → first detection accepted."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        dets = [_make_det(0, 50, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[0]["status"] == "accepted"
        assert tracks[0]["ball"] is not None


class TestLostFrames:
    def test_lost_no_detection(self, tmp_path):
        """No detection → status=lost, reason=null, raw_det=null."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        # Only frame 0, frame 1 has no detection but max_frame=1 from frame_index
        dets = [_make_det(0, 50, 50), _make_det(2, 55, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[1]["status"] == "lost"
        assert tracks[1]["reason"] is None
        assert tracks[1]["raw_det"] is None
        assert tracks[1]["ball"] is None

    def test_rejected_has_raw_det(self, tmp_path):
        """Rejected frame has raw_det with bbox/conf."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 2, "window": 3},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        dets = [_make_det(0, 50, 50, conf=0.42)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[0]["status"] == "rejected"
        assert tracks[0]["raw_det"] is not None
        assert tracks[0]["raw_det"]["conf"] == 0.42


class TestEMASmoothing:
    def test_moves_toward_measurement(self, tmp_path):
        """After EMA, position is between previous and new measurement."""
        config = {
            "tracking": {"ema_alpha": 0.5, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        # Frame 0 at x=50, frame 1 at x=100
        dets = [_make_det(0, 50, 50), _make_det(1, 100, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        # Frame 0: first, EMA init to 50
        assert tracks[0]["ball"]["x"] == 50.0
        # Frame 1: EMA = 0.5 * 100 + 0.5 * 50 = 75 (between 50 and 100)
        assert tracks[1]["ball"]["x"] == 75.0

    def test_alpha_one_instant(self, tmp_path):
        """alpha=1.0 → position equals raw measurement."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        dets = [_make_det(0, 50, 50), _make_det(1, 100, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[1]["ball"]["x"] == 100.0

    def test_alpha_zero_no_change(self, tmp_path):
        """alpha≈0 → position stays near initial."""
        config = {
            "tracking": {"ema_alpha": 0.001, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        dets = [_make_det(0, 50, 50), _make_det(1, 100, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        # Should be very close to 50 (initial)
        assert abs(tracks[1]["ball"]["x"] - 50.0) < 1.0


class TestOutputFormat:
    def test_output_has_all_fields(self, tmp_path):
        """Output has ball/status/reason/raw_det per frame."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        dets = [_make_det(0, 50, 50), _make_det(2, 55, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        events = stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        for t in tracks:
            assert "frame" in t
            assert "ball" in t
            assert "status" in t
            assert "reason" in t
            assert "raw_det" in t
            assert t["status"] in ("accepted", "rejected", "lost")

    def test_ball_only_read_works(self, tmp_path):
        """Existing consumers only read ball — verify it works."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        dets = [_make_det(0, 50, 50), _make_det(1, 55, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0)

        tracks = json.loads(tracks_path.read_text())
        # Simulate camera.py access pattern
        for t in tracks:
            ball = t.get("ball")
            if ball is not None:
                assert "x" in ball
                assert "y" in ball
                assert "confidence" in ball

    def test_events_include_jump_rejects(self, tmp_path):
        """Events list has trigger entries for rejections."""
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 10, "max_speed_px_per_s": 99999},
        }
        stab = BallStabilizer(config)

        dets = [_make_det(0, 50, 50), _make_det(1, 100, 50)]
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, dets)

        tracks_path = tmp_path / "tracks.json"
        events = stab.run(dets_path, tracks_path, fps=10.0)

        assert len(events) >= 1
        assert events[0]["trigger"] == "jump_reject"
        assert "frame" in events[0]
        assert "distance_px" in events[0]


class TestFrameRangeAndReacquisition:
    def test_tracks_len_matches_total_frames(self, tmp_path):
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 9999, "max_speed_px_per_s": 99999, "jump_max_gap_frames": 15},
        }
        stab = BallStabilizer(config)

        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, [_make_det(0, 50, 50), _make_det(2, 55, 50)])

        tracks_path = tmp_path / "tracks.json"
        stab.run(dets_path, tracks_path, fps=10.0, total_frames=8)

        tracks = json.loads(tracks_path.read_text())
        assert len(tracks) == 8
        assert tracks[-1]["frame"] == 7

    def test_reacquire_after_long_gap_accepts_far_detection(self, tmp_path):
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 50, "max_speed_px_per_s": 99999, "jump_max_gap_frames": 2},
        }
        stab = BallStabilizer(config)

        # Accepted at frame 0, long gap, then far detection at frame 4.
        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, [_make_det(0, 50, 50), _make_det(4, 250, 50)])

        tracks_path = tmp_path / "tracks.json"
        tracks_events = stab.run(dets_path, tracks_path, fps=10.0, total_frames=5)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[0]["status"] == "accepted"
        assert tracks[4]["status"] == "accepted"
        assert tracks[4]["reason"] is None
        assert all(e["frame"] != 4 for e in tracks_events)

    def test_short_gap_large_jump_still_rejected(self, tmp_path):
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 50, "max_speed_px_per_s": 99999, "jump_max_gap_frames": 10},
        }
        stab = BallStabilizer(config)

        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, [_make_det(0, 50, 50), _make_det(1, 250, 50)])

        tracks_path = tmp_path / "tracks.json"
        events = stab.run(dets_path, tracks_path, fps=10.0, total_frames=2)

        tracks = json.loads(tracks_path.read_text())
        assert tracks[1]["status"] == "rejected"
        assert tracks[1]["reason"] == "jump"
        assert any(e["trigger"] == "jump_reject" and e["frame"] == 1 for e in events)

    def test_no_detections_with_total_frames_outputs_all_lost(self, tmp_path):
        config = {
            "tracking": {"ema_alpha": 1.0, "require_persistence": 1, "window": 1},
            "filters": {"max_jump_px": 50, "max_speed_px_per_s": 99999, "jump_max_gap_frames": 10},
        }
        stab = BallStabilizer(config)

        dets_path = tmp_path / "dets.jsonl"
        _write_v1_detections(dets_path, [])

        tracks_path = tmp_path / "tracks.json"
        events = stab.run(dets_path, tracks_path, fps=10.0, total_frames=4)

        tracks = json.loads(tracks_path.read_text())
        assert len(tracks) == 4
        assert all(t["status"] == "lost" for t in tracks)
        assert all(t["ball"] is None for t in tracks)
        assert events == []
