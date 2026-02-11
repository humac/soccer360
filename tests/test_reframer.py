"""Tests for the 360-to-perspective reframer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.utils import VideoMeta


class TestVerticalFov:
    """Verify tangent-based vertical FOV computation."""

    def test_square_aspect_equal_fov(self):
        """For a square output, vFOV should equal hFOV."""
        fov_h = 90.0
        out_w, out_h = 100, 100
        fov_v = math.degrees(
            2 * math.atan(math.tan(math.radians(fov_h / 2)) * (out_h / out_w))
        )
        assert abs(fov_v - fov_h) < 1e-6

    def test_16_9_aspect(self):
        """For 16:9, vFOV should be less than hFOV and match the tangent formula."""
        fov_h = 90.0
        out_w, out_h = 1920, 1080
        fov_v = math.degrees(
            2 * math.atan(math.tan(math.radians(fov_h / 2)) * (out_h / out_w))
        )
        # 16:9 vFOV for 90 hFOV ~ 58.7 degrees (not 50.625 from linear ratio)
        assert 58.0 < fov_v < 60.0
        # Linear would give 90 * 1080/1920 = 50.625, which is wrong
        assert fov_v > 55.0

    def test_wide_fov_stays_under_180(self):
        """Even with very wide hFOV, vFOV should stay reasonable."""
        fov_h = 150.0
        out_w, out_h = 1920, 1080
        fov_v = math.degrees(
            2 * math.atan(math.tan(math.radians(fov_h / 2)) * (out_h / out_w))
        )
        assert fov_v < 180.0


class TestE2PIntegration:
    """Test py360convert e2p function directly."""

    def test_e2p_basic(self):
        """Verify py360convert produces correct output shape."""
        import py360convert

        # Create a synthetic equirectangular image (gradient)
        h, w = 320, 640
        equirect = np.zeros((h, w, 3), dtype=np.uint8)
        equirect[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)  # Red gradient

        out_h, out_w = 180, 320
        perspective = py360convert.e2p(
            equirect,
            fov_deg=(90, 50.625),
            u_deg=0,
            v_deg=0,
            out_hw=(out_h, out_w),
            mode="bilinear",
        )

        assert perspective.shape == (out_h, out_w, 3)
        assert perspective.dtype == np.uint8

    def test_e2p_different_angles(self):
        """Verify that different yaw angles produce different outputs."""
        import py360convert

        h, w = 320, 640
        equirect = np.zeros((h, w, 3), dtype=np.uint8)
        equirect[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)

        out_hw = (180, 320)

        p1 = py360convert.e2p(equirect, fov_deg=(90, 50), u_deg=0, v_deg=0, out_hw=out_hw)
        p2 = py360convert.e2p(equirect, fov_deg=(90, 50), u_deg=90, v_deg=0, out_hw=out_hw)

        # Different yaw should produce different images
        assert not np.array_equal(p1, p2)


class TestFFmpegFrameIO:
    """Test FFmpeg frame reader/writer with synthetic video."""

    def test_frame_reader(self, synthetic_video):
        from src.utils import FFmpegFrameReader

        reader = FFmpegFrameReader(
            synthetic_video,
            output_width=320,
            output_height=160,
        )

        frames = list(reader)
        assert len(frames) > 0
        assert frames[0].shape == (160, 320, 3)
        assert frames[0].dtype == np.uint8

    def test_frame_writer(self, tmp_work_dir):
        from src.utils import FFmpegFrameWriter

        output = tmp_work_dir / "test_output.mp4"
        w, h, fps = 320, 180, 10

        with FFmpegFrameWriter(output, fps, w, h, preset="ultrafast") as writer:
            for _ in range(10):
                frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
                writer.write(frame)

        assert output.exists()
        assert output.stat().st_size > 0

    def test_frame_reader_with_seek(self, synthetic_video):
        from src.utils import FFmpegFrameReader, probe_video

        meta = probe_video(synthetic_video)

        reader = FFmpegFrameReader(
            synthetic_video,
            output_width=320,
            output_height=160,
            start_frame=5,
            num_frames=5,
            fps=meta.fps,
        )

        frames = list(reader)
        assert len(frames) == 5
