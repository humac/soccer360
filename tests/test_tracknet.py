"""Tests for TrackNetV3 ball detection (non-GPU parts)."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from src.tracknet import TrackNetV3Detector

_torch_available = importlib.util.find_spec("torch") is not None


# ---------------------------------------------------------------------------
# Peak extraction tests
# ---------------------------------------------------------------------------

class TestPeakExtraction:
    """Test heatmap peak finding and sub-pixel refinement."""

    @staticmethod
    def _make_detector(threshold=0.5, radius=5):
        config = {
            "detection": {
                "ball_model": {
                    "type": "tracknet",
                    "path": None,
                    "input_height": 64,
                    "input_width": 128,
                    "heatmap_threshold": threshold,
                    "peak_radius": radius,
                },
                "device": "cpu",
            }
        }
        return TrackNetV3Detector(config)

    def test_synthetic_gaussian_center(self):
        """Gaussian peak at center of heatmap should be detected correctly."""
        det = self._make_detector(threshold=0.3)
        h, w = 64, 128
        heatmap = np.zeros((h, w), dtype=np.float32)

        cy, cx = h // 2, w // 2
        sigma = 3.0
        for y in range(h):
            for x in range(w):
                heatmap[y, x] = math.exp(
                    -((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)
                )

        result = det._extract_peak(heatmap)
        assert result is not None
        rx, ry, conf = result
        assert abs(rx - cx) < 1.0, f"Expected cx={cx}, got {rx}"
        assert abs(ry - cy) < 1.0, f"Expected cy={cy}, got {ry}"
        assert conf > 0.9

    def test_no_peak_below_threshold(self):
        """Heatmap below threshold should return None."""
        det = self._make_detector(threshold=0.8)
        heatmap = np.full((64, 128), 0.3, dtype=np.float32)
        result = det._extract_peak(heatmap)
        assert result is None

    def test_peak_at_corner(self):
        """Peak near corner should still be detected."""
        det = self._make_detector(threshold=0.3)
        h, w = 64, 128
        heatmap = np.zeros((h, w), dtype=np.float32)
        sigma = 2.0
        for y in range(h):
            for x in range(w):
                heatmap[y, x] = math.exp(
                    -((x - 5) ** 2 + (y - 5) ** 2) / (2 * sigma ** 2)
                )

        result = det._extract_peak(heatmap)
        assert result is not None
        rx, ry, conf = result
        assert abs(rx - 5) < 1.5
        assert abs(ry - 5) < 1.5

    def test_sub_pixel_accuracy(self):
        """Peak between pixels should be located with sub-pixel precision."""
        det = self._make_detector(threshold=0.3, radius=3)
        h, w = 64, 128
        heatmap = np.zeros((h, w), dtype=np.float32)

        cx_true, cy_true = 30.7, 20.3
        sigma = 2.0
        for y in range(h):
            for x in range(w):
                heatmap[y, x] = math.exp(
                    -((x - cx_true) ** 2 + (y - cy_true) ** 2)
                    / (2 * sigma ** 2)
                )

        result = det._extract_peak(heatmap)
        assert result is not None
        rx, ry, conf = result
        assert abs(rx - cx_true) < 0.5, f"Expected cx={cx_true}, got {rx}"
        assert abs(ry - cy_true) < 0.5, f"Expected cy={cy_true}, got {ry}"


# ---------------------------------------------------------------------------
# Coordinate rescaling tests
# ---------------------------------------------------------------------------

class TestRescaleCoords:
    def test_rescale_identity(self):
        det = TestPeakExtraction._make_detector()
        cx, cy = det._rescale_coords(50.0, 25.0, 128, 64)
        assert abs(cx - 50.0) < 0.01
        assert abs(cy - 25.0) < 0.01

    def test_rescale_2x(self):
        det = TestPeakExtraction._make_detector()
        cx, cy = det._rescale_coords(50.0, 25.0, 256, 128)
        assert abs(cx - 100.0) < 0.01
        assert abs(cy - 50.0) < 0.01


# ---------------------------------------------------------------------------
# Model architecture tests (require torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _torch_available, reason="torch not installed")
class TestModelArchitecture:
    """Test that the vendored model can be instantiated."""

    def test_build_model(self):
        import torch
        from src.tracknet import _build_tracknet_model

        model = _build_tracknet_model(in_channels=9)
        assert model is not None
        x = torch.randn(1, 9, 64, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 64, 128)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_preprocess_shape(self):
        """3 frames should produce (1, 9, H, W) tensor."""
        det = TestPeakExtraction._make_detector()
        frames = [
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        tensor = det._preprocess(frames)
        assert tensor.shape == (1, 9, 64, 128)


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------

class TestTrackNetConfig:
    def test_disabled_by_default(self, test_config):
        """Default config should not enable TrackNetV3."""
        det_cfg = test_config.get("detection", {}).get("ball_model", {})
        assert det_cfg.get("type") != "tracknet"

    def test_config_parsing(self):
        """TrackNetV3Detector should parse config correctly."""
        det = TestPeakExtraction._make_detector()
        assert det.input_height == 64
        assert det.input_width == 128
        assert det.heatmap_threshold == 0.5
        assert det.peak_radius == 5
