"""Tests for TrackNetV3 data utilities (heatmap generation, label conversion)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.tracknet_data import (
    bbox_to_heatmap,
    convert_yolo_labels_to_heatmaps,
    _yolo_label_to_ball_center,
)


class TestBboxToHeatmap:
    def test_center_peak(self):
        """Heatmap peak should be at the specified center."""
        hm = bbox_to_heatmap(64.0, 32.0, 64, 128, sigma=5.0)
        assert hm.shape == (64, 128)
        assert hm.dtype == np.float32
        peak_y, peak_x = np.unravel_index(hm.argmax(), hm.shape)
        assert abs(peak_x - 64) <= 1
        assert abs(peak_y - 32) <= 1
        assert abs(hm[32, 64] - 1.0) < 0.01  # peak should be ~1.0

    def test_boundary_no_overflow(self):
        """Heatmap near corner should stay in [0, 1] without overflow."""
        hm = bbox_to_heatmap(2.0, 2.0, 64, 128, sigma=3.0)
        assert hm.min() >= 0.0
        assert hm.max() <= 1.0

    def test_sigma_affects_spread(self):
        """Larger sigma should produce a wider heatmap."""
        hm_narrow = bbox_to_heatmap(64.0, 32.0, 64, 128, sigma=2.0)
        hm_wide = bbox_to_heatmap(64.0, 32.0, 64, 128, sigma=10.0)
        # Wide sigma should have more mass above 0.5
        narrow_mass = (hm_narrow > 0.5).sum()
        wide_mass = (hm_wide > 0.5).sum()
        assert wide_mass > narrow_mass


class TestYoloLabelParsing:
    def test_ball_class_parsed(self):
        """Ball class label should return center coordinates."""
        line = "32 0.5 0.5 0.05 0.1"
        result = _yolo_label_to_ball_center(line, 512, 288, ball_class=32)
        assert result is not None
        cx, cy, w, h = result
        assert abs(cx - 256.0) < 0.01
        assert abs(cy - 144.0) < 0.01
        assert abs(w - 25.6) < 0.01
        assert abs(h - 28.8) < 0.01

    def test_non_ball_class_returns_none(self):
        """Non-ball class label should return None."""
        line = "0 0.5 0.5 0.1 0.2"
        result = _yolo_label_to_ball_center(line, 512, 288, ball_class=32)
        assert result is None

    def test_malformed_line(self):
        """Malformed line should return None."""
        assert _yolo_label_to_ball_center("", 512, 288) is None
        assert _yolo_label_to_ball_center("32", 512, 288) is None


class TestConvertLabels:
    def test_convert_creates_heatmaps(self):
        """Label conversion should produce .npy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_dir = Path(tmpdir) / "labels"
            labels_dir.mkdir()
            output_dir = Path(tmpdir) / "heatmaps"

            # Create a label file with one ball annotation
            (labels_dir / "frame_0001.txt").write_text("32 0.5 0.5 0.03 0.06\n")
            # Create a label file with no ball
            (labels_dir / "frame_0002.txt").write_text("0 0.5 0.5 0.1 0.2\n")

            count = convert_yolo_labels_to_heatmaps(
                labels_dir=labels_dir,
                output_dir=output_dir,
                img_h=64,
                img_w=128,
                ball_class=32,
            )

            assert count == 2
            assert (output_dir / "frame_0001.npy").exists()
            assert (output_dir / "frame_0002.npy").exists()

            # First heatmap should have a peak
            hm1 = np.load(output_dir / "frame_0001.npy")
            assert hm1.max() > 0.9

            # Second heatmap should be all zeros (no ball)
            hm2 = np.load(output_dir / "frame_0002.npy")
            assert hm2.max() < 0.01
