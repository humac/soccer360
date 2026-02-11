"""Tests for the 360-to-perspective reframer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import src.reframer as reframer_mod
from src.reframer import Reframer
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


class TestReframerHardening:
    def test_render_broadcast_rejects_zero_frames(self, test_config, tmp_path):
        reframer = Reframer(test_config)
        camera_path = tmp_path / "camera_path.json"
        camera_path.write_text("[]")

        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=0.0, total_frames=0, codec="h264",
        )
        with pytest.raises(ValueError, match="camera_path has no frames"):
            reframer.render_broadcast(
                video_path="dummy.mp4",
                meta=meta,
                camera_path_file=camera_path,
                output_path=tmp_path / "broadcast.mp4",
            )

    def test_render_tactical_rejects_zero_frames(self, test_config, tmp_path):
        reframer = Reframer(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=0.0, total_frames=0, codec="h264",
        )
        with pytest.raises(ValueError, match="source has no frames"):
            reframer.render_tactical(
                video_path="dummy.mp4",
                meta=meta,
                output_path=tmp_path / "tactical.mp4",
            )

    def test_short_render_segment_raises(self, monkeypatch, tmp_path):
        class DummyReader:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                frame = np.zeros((8, 8, 3), dtype=np.uint8)
                yield frame
                yield frame

        class DummyWriter:
            def __init__(self, *args, **kwargs):
                self.writes = 0

            def write(self, frame):
                self.writes += 1

            def close(self):
                return None

        monkeypatch.setattr(reframer_mod, "FFmpegFrameReader", DummyReader)
        monkeypatch.setattr(reframer_mod, "FFmpegFrameWriter", DummyWriter)
        monkeypatch.setattr(reframer_mod.py360convert, "e2p", lambda frame, **kwargs: frame)

        camera_entries = [{"yaw": 0.0, "pitch": 0.0, "fov": 90.0} for _ in range(3)]
        with pytest.raises(RuntimeError, match="Short render"):
            reframer_mod._render_segment(
                video_path="dummy.mp4",
                camera_entries=camera_entries,
                start_frame=0,
                fps=30.0,
                source_w=8,
                source_h=8,
                out_w=8,
                out_h=8,
                codec="libx264",
                crf=23,
                preset="ultrafast",
                output_path=str(tmp_path / "seg.mp4"),
                overlap_frames=0,
            )

    def test_tactical_cleanup_on_worker_failure(self, test_config, monkeypatch, tmp_path):
        class DummyFuture:
            def __init__(self, fail: bool = False):
                self.fail = fail

            def result(self):
                if self.fail:
                    raise RuntimeError("worker failed")
                return None

        class DummyPool:
            def __init__(self, max_workers):
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args):
                self.calls += 1
                seg_path = Path(args[14])
                seg_path.parent.mkdir(parents=True, exist_ok=True)
                seg_path.write_bytes(b"segment")
                return DummyFuture(fail=self.calls == 2)

        monkeypatch.setattr(reframer_mod, "ProcessPoolExecutor", DummyPool)
        monkeypatch.setattr(reframer_mod, "concat_segments", lambda *args, **kwargs: None)

        reframer = Reframer(test_config)
        meta = VideoMeta(
            width=640, height=320, fps=30.0,
            duration=0.13, total_frames=4, codec="h264",
        )

        with pytest.raises(RuntimeError, match="worker failed"):
            reframer.render_tactical(
                video_path="dummy.mp4",
                meta=meta,
                output_path=tmp_path / "tactical.mp4",
            )

        assert list(tmp_path.glob("tactical_seg_*.mp4")) == []
