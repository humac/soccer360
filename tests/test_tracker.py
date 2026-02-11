"""Tests for ByteTrack ball tracker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tracker import ByteTrackInstance, Tracker, iou_batch


class TestIouBatch:
    def test_identical_boxes(self):
        a = np.array([[0, 0, 10, 10]])
        result = iou_batch(a, a)
        assert abs(result[0, 0] - 1.0) < 1e-6

    def test_no_overlap(self):
        a = np.array([[0, 0, 5, 5]])
        b = np.array([[10, 10, 20, 20]])
        result = iou_batch(a, b)
        assert abs(result[0, 0]) < 1e-6

    def test_partial_overlap(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[5, 5, 15, 15]])
        result = iou_batch(a, b)
        # Intersection = 5x5 = 25, Union = 100 + 100 - 25 = 175
        expected = 25.0 / 175.0
        assert abs(result[0, 0] - expected) < 1e-4

    def test_multiple_boxes(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        b = np.array([[5, 5, 15, 15]])
        result = iou_batch(a, b)
        assert result.shape == (2, 1)
        assert result[0, 0] > 0  # partial overlap with first
        assert abs(result[1, 0]) < 1e-6  # no overlap with second


class TestByteTrackInstance:
    def test_creates_track(self):
        bt = ByteTrackInstance()
        dets = np.array([[10, 10, 20, 20, 0.9]])
        result = bt.update(dets)
        assert len(result) == 1
        assert "track_id" in result[0]

    def test_score_reflects_detection_confidence(self):
        bt = ByteTrackInstance()

        # High confidence detection
        dets = np.array([[10, 10, 20, 20, 0.95]])
        result = bt.update(dets)
        assert len(result) == 1
        # Score should be dominated by det confidence (0.7 weight)
        # First frame: streak=1, streak_factor=0.2, score ~= 0.7*0.95 + 0.3*0.2 = 0.725
        assert result[0]["score"] > 0.6

        # Low confidence detection
        bt2 = ByteTrackInstance()
        dets_low = np.array([[10, 10, 20, 20, 0.15]])
        result_low = bt2.update(dets_low)
        assert len(result_low) == 1
        assert result_low[0]["score"] < result[0]["score"]

    def test_maintains_track(self):
        bt = ByteTrackInstance()

        # Frame 1
        dets1 = np.array([[10, 10, 20, 20, 0.9]])
        r1 = bt.update(dets1)
        track_id = r1[0]["track_id"]

        # Frame 2 (slightly moved)
        dets2 = np.array([[12, 10, 22, 20, 0.9]])
        r2 = bt.update(dets2)
        assert len(r2) == 1
        assert r2[0]["track_id"] == track_id

    def test_handles_no_detections(self):
        bt = ByteTrackInstance()
        dets = np.empty((0, 5))
        result = bt.update(dets)
        assert len(result) == 0

    def test_recovers_from_lost(self):
        bt = ByteTrackInstance(track_buffer=10)

        # Frame 1: ball detected
        bt.update(np.array([[50, 50, 60, 60, 0.9]]))

        # Frames 2-4: ball lost
        for _ in range(3):
            bt.update(np.empty((0, 5)))

        # Frame 5: ball reappears nearby
        result = bt.update(np.array([[55, 50, 65, 60, 0.9]]))
        assert len(result) >= 1


class TestTracker:
    def test_run_produces_tracks(self, test_config, sample_detections, tmp_work_dir):
        tracker = Tracker(test_config)
        output = tmp_work_dir / "tracks.json"
        tracker.run(sample_detections, output)

        assert output.exists()
        with open(output) as f:
            tracks = json.load(f)

        assert len(tracks) == 30
        # Most frames should have ball detected
        ball_found = sum(1 for t in tracks if t["ball"] is not None)
        assert ball_found >= 20

    def test_ball_selection(self, test_config, sample_detections, tmp_work_dir):
        tracker = Tracker(test_config)
        output = tmp_work_dir / "tracks.json"
        tracker.run(sample_detections, output)

        with open(output) as f:
            tracks = json.load(f)

        for t in tracks:
            if t["ball"] is not None:
                assert "x" in t["ball"]
                assert "y" in t["ball"]
                assert "track_id" in t["ball"]
