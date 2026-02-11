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
    }
    return cfg


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
