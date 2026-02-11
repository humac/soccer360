"""Tests for the detector module (non-GPU parts)."""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import Detector


class TestNMS:
    def test_no_overlap(self):
        boxes = np.array([[0, 0, 5, 5], [10, 10, 15, 15]])
        scores = np.array([0.9, 0.8])
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2

    def test_full_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]])
        scores = np.array([0.9, 0.8])
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 1
        assert keep[0] == 0  # highest score kept

    def test_partial_overlap_below_threshold(self):
        boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20]])
        scores = np.array([0.9, 0.8])
        # IoU = 25/275 ~ 0.09, below 0.5 threshold
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 2

    def test_empty(self):
        boxes = np.empty((0, 4))
        scores = np.empty(0)
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert len(keep) == 0

    def test_single_box(self):
        boxes = np.array([[0, 0, 10, 10]])
        scores = np.array([0.9])
        keep = Detector._nms(boxes, scores, iou_threshold=0.5)
        assert keep == [0]
