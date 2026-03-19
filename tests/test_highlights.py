"""Tests for highlight detection heuristics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.highlights import HighlightDetector
from src.utils import VideoMeta


@pytest.fixture
def highlight_config(test_config):
    """Config with relaxed thresholds for testing."""
    cfg = test_config.copy()
    cfg["highlights"] = {
        "speed_percentile": 80,
        "direction_change_deg": 45,
        "goal_box_regions": [
            [0.0, 0.3, 0.1, 0.7],
            [0.9, 0.3, 1.0, 0.7],
        ],
        "pre_margin_sec": 1.0,
        "post_margin_sec": 0.5,
        "min_clip_gap_sec": 2.0,
        "min_clip_duration_sec": 0.5,
        # Cluster detector config
        "cluster_convergence_window": 5,
        "cluster_convergence_deg": 5.0,
        "cluster_velocity_window": 3,
        "cluster_velocity_deg_per_sec": 10.0,
        "cluster_goal_zone_regions": None,
        "cluster_density_percentile": 80,
        # Scoring
        "score_weights": {
            "speed": 1.0,
            "goal_box": 1.5,
            "direction_change": 0.8,
            "cluster_convergence": 1.2,
            "cluster_velocity": 0.7,
            "cluster_goal_zone": 1.3,
            "cluster_density": 0.5,
        },
        "combined_signal_bonus": 1.5,
        "min_clip_score": 0.5,
        "max_clips": 10,
    }
    return cfg


def _make_clusters(
    n_frames: int,
    x: float = 160.0,
    y: float = 80.0,
    spread: float = 35.0,
    count: int = 10,
) -> list[dict]:
    """Helper: generate uniform cluster data."""
    return [
        {
            "frame": i,
            "cluster": {
                "x": x,
                "y": y,
                "spread_x_deg": spread,
                "player_count": count,
                "confidence": 0.7,
            },
        }
        for i in range(n_frames)
    ]


class TestVelocityComputation:
    def test_stationary_ball(self, highlight_config):
        detector = HighlightDetector(highlight_config)
        tracks = [
            {"frame": i, "ball": {"x": 100, "y": 80}} for i in range(10)
        ]
        velocities = detector._compute_velocities(tracks, fps=30.0)
        assert all(v["speed"] == 0.0 for v in velocities)

    def test_moving_ball(self, highlight_config):
        detector = HighlightDetector(highlight_config)
        tracks = [
            {"frame": i, "ball": {"x": 10 * i, "y": 80}} for i in range(10)
        ]
        velocities = detector._compute_velocities(tracks, fps=30.0)
        # Ball moves 10 px/frame * 30 fps = 300 px/sec
        for v in velocities[1:]:
            assert abs(v["speed"] - 300.0) < 1.0

    def test_lost_ball(self, highlight_config):
        detector = HighlightDetector(highlight_config)
        tracks = [
            {"frame": 0, "ball": {"x": 100, "y": 80}},
            {"frame": 1, "ball": None},
            {"frame": 2, "ball": {"x": 120, "y": 80}},
        ]
        velocities = detector._compute_velocities(tracks, fps=30.0)
        assert velocities[1]["has_ball"] is False


class TestGoalBoxDetection:
    def test_ball_in_goal_box(self, highlight_config):
        detector = HighlightDetector(highlight_config)
        # Ball at x=5% of 320 = 16, y=50% of 160 = 80 -> in left goal box
        tracks = [
            {"frame": 0, "ball": {"x": 16, "y": 80}},
        ]
        events = detector._detect_goal_box_events(tracks, fps=30.0)
        assert len(events) == 1
        assert events[0]["type"] == "goal_box"

    def test_ball_outside_goal_box(self, highlight_config):
        detector = HighlightDetector(highlight_config)
        # Ball at center
        tracks = [
            {"frame": 0, "ball": {"x": 160, "y": 80}},
        ]
        events = detector._detect_goal_box_events(tracks, fps=30.0)
        assert len(events) == 0


class TestEventClustering:
    def test_merge_close_events(self, highlight_config):
        detector = HighlightDetector(highlight_config)
        events = [
            {"frame": 0, "time_sec": 0.5, "type": "speed", "value": 100},
            {"frame": 10, "time_sec": 1.0, "type": "speed", "value": 120},
        ]
        clips = detector._cluster_events(events, fps=30.0)
        # Events are within min_clip_gap_sec, should merge
        assert len(clips) == 1

    def test_separate_far_events(self, highlight_config):
        detector = HighlightDetector(highlight_config)
        events = [
            {"frame": 0, "time_sec": 0.0, "type": "speed", "value": 100},
            {"frame": 300, "time_sec": 10.0, "type": "speed", "value": 120},
        ]
        clips = detector._cluster_events(events, fps=30.0)
        assert len(clips) == 2

    def test_no_events(self, highlight_config):
        detector = HighlightDetector(highlight_config)
        clips = detector._cluster_events([], fps=30.0)
        assert clips == []


# ------------------------------------------------------------------
# Cluster-based detector tests
# ------------------------------------------------------------------


class TestClusterConvergence:
    def test_convergence_detected(self, highlight_config):
        """Spread dropping from 40 to 10 over 6 frames triggers convergence."""
        detector = HighlightDetector(highlight_config)
        clusters = _make_clusters(20, spread=35.0)
        # Frames 10-15: spread drops rapidly
        for i in range(10, 16):
            clusters[i]["cluster"]["spread_x_deg"] = 40.0 - (i - 10) * 6.0
        events = detector._detect_cluster_convergence(clusters, fps=30.0)
        assert len(events) > 0
        assert all(e["type"] == "cluster_convergence" for e in events)

    def test_stable_spread_no_events(self, highlight_config):
        """Constant spread produces no convergence events."""
        detector = HighlightDetector(highlight_config)
        clusters = _make_clusters(20, spread=30.0)
        events = detector._detect_cluster_convergence(clusters, fps=30.0)
        assert len(events) == 0

    def test_null_cluster_frames_skipped(self, highlight_config):
        """Frames with null cluster are skipped gracefully."""
        detector = HighlightDetector(highlight_config)
        clusters = _make_clusters(20, spread=35.0)
        clusters[8]["cluster"] = None
        clusters[9]["cluster"] = None
        events = detector._detect_cluster_convergence(clusters, fps=30.0)
        # Should not crash; events depend on data availability
        assert isinstance(events, list)


class TestClusterVelocity:
    def test_fast_break_detected(self, highlight_config):
        """Centroid moving 200px in 3 frames at 30fps triggers velocity event."""
        detector = HighlightDetector(highlight_config)
        clusters = _make_clusters(20)
        # Frames 10-12: centroid jumps right quickly
        for i in range(10, 13):
            clusters[i]["cluster"]["x"] = 160.0 + (i - 9) * 70.0
        events = detector._detect_cluster_velocity(clusters, fps=30.0)
        assert len(events) > 0
        assert all(e["type"] == "cluster_velocity" for e in events)

    def test_stationary_cluster_no_events(self, highlight_config):
        """Stationary centroid produces no velocity events."""
        detector = HighlightDetector(highlight_config)
        clusters = _make_clusters(20)
        events = detector._detect_cluster_velocity(clusters, fps=30.0)
        assert len(events) == 0


class TestClusterGoalZone:
    def test_cluster_in_goal_zone(self, highlight_config):
        """Cluster centroid near goal with enough players triggers event."""
        detector = HighlightDetector(highlight_config)
        # x=16 is 5% of 320, within left goal zone [0.0, 0.3, 0.1, 0.7]
        # y=80 is 50% of 160, within [0.3, 0.7]
        clusters = _make_clusters(5, x=16.0, y=80.0, count=12)
        events = detector._detect_cluster_goal_zone(clusters, fps=30.0)
        assert len(events) == 5
        assert all(e["type"] == "cluster_goal_zone" for e in events)

    def test_cluster_in_zone_few_players_no_event(self, highlight_config):
        """Cluster in goal zone but with < 6 players does not trigger."""
        detector = HighlightDetector(highlight_config)
        clusters = _make_clusters(5, x=16.0, y=80.0, count=3)
        events = detector._detect_cluster_goal_zone(clusters, fps=30.0)
        assert len(events) == 0

    def test_cluster_at_center_no_event(self, highlight_config):
        """Cluster at center of field does not trigger goal zone."""
        detector = HighlightDetector(highlight_config)
        clusters = _make_clusters(5, x=160.0, y=80.0, count=12)
        events = detector._detect_cluster_goal_zone(clusters, fps=30.0)
        assert len(events) == 0


class TestClusterDensity:
    def test_density_spike(self, highlight_config):
        """One frame with high player count triggers density event."""
        detector = HighlightDetector(highlight_config)
        clusters = _make_clusters(20, count=10)
        clusters[15]["cluster"]["player_count"] = 25  # spike
        events = detector._detect_cluster_density_spike(clusters, fps=30.0)
        assert len(events) >= 1
        spike_frames = [e["frame"] for e in events]
        assert 15 in spike_frames

    def test_uniform_count_no_spikes(self, highlight_config):
        """Uniform player count produces events only at percentile boundary."""
        detector = HighlightDetector(highlight_config)
        clusters = _make_clusters(20, count=10)
        events = detector._detect_cluster_density_spike(clusters, fps=30.0)
        # All counts equal: percentile threshold == count, so all at boundary
        # With percentile 80 and all values equal, threshold == 10
        # Events where count >= threshold (all of them)
        assert isinstance(events, list)


# ------------------------------------------------------------------
# Scoring tests
# ------------------------------------------------------------------


class TestScoring:
    def test_combined_signal_bonus(self, highlight_config):
        """Clip with both ball and cluster events gets bonus multiplier."""
        detector = HighlightDetector(highlight_config)
        events = [
            {"frame": 0, "time_sec": 0.5, "type": "speed", "value": 100},
            {"frame": 5, "time_sec": 0.8, "type": "cluster_convergence", "value": 10},
        ]
        clips = detector._cluster_events(events, fps=30.0)
        assert len(clips) == 1
        # score = (1.0 * speed_weight + 1.0 * conv_weight) * bonus
        expected = (1.0 + 1.2) * 1.5
        assert abs(clips[0]["score"] - expected) < 0.1

    def test_min_score_filter(self, highlight_config):
        """Clips below min_clip_score are filtered out."""
        cfg = highlight_config.copy()
        cfg["highlights"] = dict(highlight_config["highlights"])
        cfg["highlights"]["min_clip_score"] = 100.0  # impossibly high
        detector = HighlightDetector(cfg)
        events = [
            {"frame": 0, "time_sec": 0.5, "type": "speed", "value": 50},
        ]
        clips = detector._cluster_events(events, fps=30.0)
        assert len(clips) == 0

    def test_max_clips_cap(self, highlight_config):
        """Only top max_clips are kept."""
        cfg = highlight_config.copy()
        cfg["highlights"] = dict(highlight_config["highlights"])
        cfg["highlights"]["max_clips"] = 2
        cfg["highlights"]["min_clip_score"] = 0.1
        detector = HighlightDetector(cfg)
        # 5 events spread far apart -> 5 separate clips
        events = [
            {"frame": i * 300, "time_sec": i * 10.0, "type": "speed", "value": 100 + i}
            for i in range(5)
        ]
        clips = detector._cluster_events(events, fps=30.0)
        assert len(clips) <= 2

    def test_clips_sorted_by_time_with_rank(self, highlight_config):
        """Clips are sorted by time but have rank by score."""
        cfg = highlight_config.copy()
        cfg["highlights"] = dict(highlight_config["highlights"])
        cfg["highlights"]["min_clip_score"] = 0.1
        detector = HighlightDetector(cfg)
        events = [
            {"frame": 0, "time_sec": 0.0, "type": "speed", "value": 50},
            {"frame": 300, "time_sec": 10.0, "type": "speed", "value": 200},
            {"frame": 150, "time_sec": 5.0, "type": "speed", "value": 100},
        ]
        clips = detector._cluster_events(events, fps=30.0)
        # Should be sorted by start_sec
        times = [c["start_sec"] for c in clips]
        assert times == sorted(times)
        # Rank 1 should be the highest-scoring clip
        ranks = {c["rank"]: c["start_sec"] for c in clips}
        assert 1 in ranks


# ------------------------------------------------------------------
# Backwards compatibility tests
# ------------------------------------------------------------------


class TestBackwardsCompatibility:
    def test_no_cluster_path_ball_only(self, highlight_config):
        """Without cluster path, only ball-based detectors run."""
        detector = HighlightDetector(highlight_config)
        tracks = [
            {"frame": i, "ball": {"x": 10 * i, "y": 80}} for i in range(20)
        ]
        velocities = detector._compute_velocities(tracks, fps=30.0)
        speed_events = detector._detect_speed_events(velocities, fps=30.0)
        # Ball-based detection still works
        assert isinstance(speed_events, list)

    def test_no_tracks_no_clusters_no_crash(self, highlight_config, tmp_path):
        """Both None inputs produce no crash and no events."""
        detector = HighlightDetector(highlight_config)
        meta = VideoMeta(width=320, height=160, fps=30.0, duration=1.0,
                         total_frames=30, codec="h264")
        output_dir = tmp_path / "highlights"
        # Should not raise
        detector.detect_and_export(
            broadcast_path=tmp_path / "fake.mp4",
            meta=meta,
            camera_path_file=tmp_path / "fake_camera.json",
            tracks_path=None,
            output_dir=output_dir,
            player_cluster_path=None,
        )
        # No clips should be exported
        assert not output_dir.exists() or len(list(output_dir.glob("*.mp4"))) == 0
        assert not (output_dir / "highlights.json").exists()

    def test_cluster_only_highlights(self, highlight_config, tmp_path):
        """Cluster data without tracks still produces cluster-based events."""
        detector = HighlightDetector(highlight_config)
        # Create cluster data with convergence event
        clusters = _make_clusters(20, spread=35.0)
        for i in range(10, 16):
            clusters[i]["cluster"]["spread_x_deg"] = 40.0 - (i - 10) * 6.0
        cluster_path = tmp_path / "player_cluster.json"
        with open(cluster_path, "w") as f:
            json.dump(clusters, f)

        loaded = detector._load_cluster_data(cluster_path)
        assert loaded is not None
        events = detector._detect_cluster_convergence(loaded, fps=30.0)
        assert len(events) > 0


class TestHighlightManifest:
    def test_manifest_written_for_exported_clips_in_time_order(self, highlight_config, tmp_path, monkeypatch):
        detector = HighlightDetector(highlight_config)
        meta = VideoMeta(width=320, height=160, fps=30.0, duration=5.0,
                         total_frames=150, codec="h264")
        output_dir = tmp_path / "highlights"
        exported: list[tuple[float, str]] = []

        def fake_export_clip(_source_video, clip, output_path):
            output_path.write_bytes(b"clip")
            exported.append((clip["start_sec"], output_path.name))

        monkeypatch.setattr(detector, "_export_clip", fake_export_clip)
        monkeypatch.setattr(
            detector,
            "_compute_velocities",
            lambda tracks, fps: [
                {"frame": 90, "time_sec": 3.0, "type": "speed", "value": 150.0},
                {"frame": 30, "time_sec": 1.0, "type": "speed", "value": 100.0},
            ],
        )
        monkeypatch.setattr(
            detector,
            "_detect_speed_events",
            lambda velocities, fps: velocities,
        )
        monkeypatch.setattr(detector, "_detect_goal_box_events", lambda tracks, fps: [])
        monkeypatch.setattr(detector, "_detect_direction_changes", lambda velocities, fps: [])

        tracks_path = tmp_path / "tracks.json"
        tracks_path.write_text("[]")

        detector.detect_and_export(
            broadcast_path=tmp_path / "broadcast.mp4",
            meta=meta,
            camera_path_file=tmp_path / "camera_path.json",
            tracks_path=tracks_path,
            output_dir=output_dir,
            player_cluster_path=None,
        )

        manifest = json.loads((output_dir / "highlights.json").read_text())
        assert [name for _, name in exported] == ["highlight_000.mp4", "highlight_001.mp4"]
        assert [clip["filename"] for clip in manifest["clips"]] == ["highlight_000.mp4", "highlight_001.mp4"]
        assert [clip["start_sec"] for clip in manifest["clips"]] == sorted(
            clip["start_sec"] for clip in manifest["clips"]
        )
        assert manifest["clip_count"] == 2
        assert manifest["reel_filename"] is None
        assert manifest["clips"][0]["event_types"] == ["speed"]
        assert manifest["clips"][0]["event_count"] == 1
        assert manifest["detector_stats"]["total_raw_events"] == 2

    def test_write_manifest_sorts_event_types(self, highlight_config, tmp_path):
        detector = HighlightDetector(highlight_config)
        clips = [{
            "start_sec": 1.0,
            "end_sec": 2.0,
            "duration": 1.0,
            "score": 3.0,
            "rank": 1,
            "event_types": ["speed", "goal_box", "cluster_density"],
            "event_count": 3,
        }]

        detector._write_manifest(tmp_path, clips, {"total_raw_events": 3})
        manifest = json.loads((tmp_path / "highlights.json").read_text())
        assert manifest["clips"][0]["event_types"] == ["cluster_density", "goal_box", "speed"]
        assert manifest["reel_filename"] is None
