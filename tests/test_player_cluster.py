"""Tests for center-of-play player cluster computation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.player_cluster import PlayerClusterComputer


@pytest.fixture
def cop_config(test_config):
    """Config with center_of_play enabled."""
    return test_config


@pytest.fixture
def cluster_computer(cop_config):
    return PlayerClusterComputer(cop_config)


def _write_detections(path: Path, detections: list[dict]):
    with open(path, "w") as f:
        for d in detections:
            f.write(json.dumps(d) + "\n")


def _make_player_det(frame: int, cx: float, cy: float, conf: float = 0.6) -> dict:
    """Create a player detection dict."""
    return {
        "frame_index": frame,
        "bbox_xyxy": [cx - 10, cy - 20, cx + 10, cy + 20],
        "conf": conf,
        "class_id": 0,
    }


def _make_ball_det(frame: int, cx: float, cy: float, conf: float = 0.8) -> dict:
    """Create a ball detection dict."""
    return {
        "frame_index": frame,
        "bbox_xyxy": [cx - 3, cy - 3, cx + 3, cy + 3],
        "conf": conf,
        "class_id": 32,
    }


class TestPlayerClusterComputer:
    def test_basic_cluster(self, cluster_computer, tmp_path):
        """Players evenly spread across frame produce a centered cluster."""
        dets = []
        for frame in range(10):
            # 10 players spread across x=[50..270], cy=80
            for i in range(10):
                cx = 50 + i * (220 / 9)
                dets.append(_make_player_det(frame, cx, 80.0))
        det_path = tmp_path / "detections.jsonl"
        _write_detections(det_path, dets)

        out_path = tmp_path / "player_cluster.json"
        cluster_computer.run(det_path, out_path, total_frames=10)

        result = json.loads(out_path.read_text())
        assert len(result) == 10

        # All frames should have valid clusters
        for entry in result:
            assert entry["cluster"] is not None
            cl = entry["cluster"]
            assert cl["player_count"] == 10
            # Centroid should be near center of spread
            assert 100 < cl["x"] < 220
            assert cl["spread_x_deg"] > 0

    def test_too_few_players(self, cluster_computer, tmp_path):
        """Frames with fewer than min_players produce null cluster."""
        dets = []
        # Frame 0: only 2 players (below min_players=4)
        for i in range(2):
            dets.append(_make_player_det(0, 100.0 + i * 50, 80.0))
        # Frame 1: 5 players (above min)
        for i in range(5):
            dets.append(_make_player_det(1, 100.0 + i * 30, 80.0))

        det_path = tmp_path / "detections.jsonl"
        _write_detections(det_path, dets)

        out_path = tmp_path / "player_cluster.json"
        cluster_computer.run(det_path, out_path, total_frames=2)

        result = json.loads(out_path.read_text())
        assert result[0]["cluster"] is None
        assert result[1]["cluster"] is not None

    def test_trimmed_mean_excludes_outliers(self, cluster_computer, tmp_path):
        """Outlier players (GKs at extremes) should be trimmed from centroid."""
        dets = []
        # 12 players: 10 clustered at center, 1 GK at x=5, 1 GK at x=315
        for i in range(10):
            dets.append(_make_player_det(0, 140.0 + i * 5, 80.0))
        dets.append(_make_player_det(0, 5.0, 80.0))   # GK left
        dets.append(_make_player_det(0, 315.0, 80.0))  # GK right

        det_path = tmp_path / "detections.jsonl"
        _write_detections(det_path, dets)

        out_path = tmp_path / "player_cluster.json"
        cluster_computer.run(det_path, out_path, total_frames=1)

        result = json.loads(out_path.read_text())
        cl = result[0]["cluster"]
        assert cl is not None
        # Centroid should be near 162.5 (center of the 10 clustered players)
        # not pulled toward the extreme GKs
        assert 130 < cl["x"] < 200

    def test_low_confidence_filtered(self, cluster_computer, tmp_path):
        """Player detections below min_player_conf are excluded."""
        dets = []
        # 5 players above threshold
        for i in range(5):
            dets.append(_make_player_det(0, 100.0 + i * 30, 80.0, conf=0.5))
        # 5 players below threshold (min_player_conf=0.30)
        for i in range(5):
            dets.append(_make_player_det(0, 100.0 + i * 30, 80.0, conf=0.10))

        det_path = tmp_path / "detections.jsonl"
        _write_detections(det_path, dets)

        out_path = tmp_path / "player_cluster.json"
        cluster_computer.run(det_path, out_path, total_frames=1)

        result = json.loads(out_path.read_text())
        cl = result[0]["cluster"]
        assert cl is not None
        assert cl["player_count"] == 5  # only high-conf counted

    def test_ball_detections_ignored(self, cluster_computer, tmp_path):
        """Ball detections (class 32) are not included in player cluster."""
        dets = []
        # 5 players
        for i in range(5):
            dets.append(_make_player_det(0, 100.0 + i * 30, 80.0))
        # 1 ball
        dets.append(_make_ball_det(0, 160.0, 80.0))

        det_path = tmp_path / "detections.jsonl"
        _write_detections(det_path, dets)

        out_path = tmp_path / "player_cluster.json"
        cluster_computer.run(det_path, out_path, total_frames=1)

        result = json.loads(out_path.read_text())
        cl = result[0]["cluster"]
        assert cl["player_count"] == 5  # ball not counted

    def test_ema_smoothing(self, cluster_computer, tmp_path):
        """EMA smoothing reduces frame-to-frame jitter in cluster position."""
        dets = []
        # Frame 0-4: players centered at x=100
        # Frame 5-9: players jump to x=200
        for frame in range(10):
            cx = 100.0 if frame < 5 else 200.0
            for i in range(6):
                dets.append(_make_player_det(frame, cx + i * 5, 80.0))

        det_path = tmp_path / "detections.jsonl"
        _write_detections(det_path, dets)

        out_path = tmp_path / "player_cluster.json"
        cluster_computer.run(det_path, out_path, total_frames=10)

        result = json.loads(out_path.read_text())
        # After jump at frame 5, cluster x should not immediately reach 200
        # due to EMA smoothing (alpha=0.2 means slow response)
        cl_5 = result[5]["cluster"]
        assert cl_5["x"] < 180  # should be smoothed, not yet at 200

    def test_empty_detections(self, cluster_computer, tmp_path):
        """Empty detections file produces all-null clusters."""
        det_path = tmp_path / "detections.jsonl"
        det_path.write_text("")

        out_path = tmp_path / "player_cluster.json"
        cluster_computer.run(det_path, out_path, total_frames=5)

        result = json.loads(out_path.read_text())
        assert len(result) == 5
        for entry in result:
            assert entry["cluster"] is None

    def test_ema_carryforward_on_gap(self, cluster_computer, tmp_path):
        """When players disappear for a frame, EMA carries forward last position."""
        dets = []
        # Frame 0: 6 players at x=150
        for i in range(6):
            dets.append(_make_player_det(0, 150.0 + i * 5, 80.0))
        # Frame 1: no players (gap)
        # Frame 2: 6 players at x=150
        for i in range(6):
            dets.append(_make_player_det(2, 150.0 + i * 5, 80.0))

        det_path = tmp_path / "detections.jsonl"
        _write_detections(det_path, dets)

        out_path = tmp_path / "player_cluster.json"
        cluster_computer.run(det_path, out_path, total_frames=3)

        result = json.loads(out_path.read_text())
        # Frame 1 should carry forward from frame 0 (not null)
        assert result[1]["cluster"] is not None
        assert result[1]["cluster"]["player_count"] == 0  # no actual players
        assert result[1]["cluster"]["x"] > 0  # carried forward position
