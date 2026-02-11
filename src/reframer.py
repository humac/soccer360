"""360-to-perspective video reframing with parallel segment rendering.

Reads equirectangular frames, applies py360convert e2p with per-frame
camera parameters, and encodes the perspective output.

Broadcast view: tracks ball via camera_path.json
Tactical view: fixed wide-angle centered on field
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import py360convert

from .utils import (
    FFmpegFrameReader,
    FFmpegFrameWriter,
    VideoMeta,
    concat_segments,
    load_json,
)

logger = logging.getLogger("soccer360.reframer")


def _render_segment(
    video_path: str,
    camera_entries: list[dict],
    start_frame: int,
    fps: float,
    source_w: int | None,
    source_h: int | None,
    out_w: int,
    out_h: int,
    codec: str,
    crf: int,
    preset: str,
    output_path: str,
    overlap_frames: int = 0,
):
    """Render a range of frames as a video segment. Runs in worker process.

    This is a module-level function (not a method) so it can be pickled
    for multiprocessing. source_w/source_h = None means native resolution.

    overlap_frames: extra frames rendered before the real content to let the
    codec warm up; these are decoded and projected but not written to output.
    """
    actual_start = max(0, start_frame - overlap_frames)
    skip_count = start_frame - actual_start
    total_read = skip_count + len(camera_entries)

    # Build camera entries for overlap frames (repeat first entry's params)
    if skip_count > 0:
        overlap_cam = [camera_entries[0]] * skip_count
        full_cam = overlap_cam + camera_entries
    else:
        full_cam = camera_entries

    reader = FFmpegFrameReader(
        video_path,
        output_width=source_w,
        output_height=source_h,
        start_frame=actual_start,
        num_frames=total_read,
        fps=fps,
    )

    writer = FFmpegFrameWriter(
        output_path, fps,
        width=out_w, height=out_h,
        codec=codec, crf=crf, preset=preset,
    )

    try:
        for i, (frame, cam) in enumerate(zip(reader, full_cam)):
            fov_h = cam["fov"]
            fov_v = math.degrees(
                2 * math.atan(math.tan(math.radians(fov_h / 2)) * (out_h / out_w))
            )

            perspective = py360convert.e2p(
                frame,
                fov_deg=(fov_h, fov_v),
                u_deg=cam["yaw"],
                v_deg=cam["pitch"],
                out_hw=(out_h, out_w),
                mode="bilinear",
            )

            # Skip overlap warmup frames
            if i >= skip_count:
                writer.write(perspective)
    finally:
        writer.close()


def _render_tactical_segment(
    video_path: str,
    start_frame: int,
    num_frames: int,
    fps: float,
    source_w: int | None,
    source_h: int | None,
    out_w: int,
    out_h: int,
    tactical_yaw: float,
    tactical_pitch: float,
    tactical_fov: float,
    codec: str,
    crf: int,
    preset: str,
    output_path: str,
    overlap_frames: int = 0,
):
    """Render tactical wide-angle segment with fixed camera. Worker process.

    overlap_frames: extra frames at segment start for codec warmup (not written).
    """
    actual_start = max(0, start_frame - overlap_frames)
    skip_count = start_frame - actual_start
    total_read = skip_count + num_frames

    reader = FFmpegFrameReader(
        video_path,
        output_width=source_w,
        output_height=source_h,
        start_frame=actual_start,
        num_frames=total_read,
        fps=fps,
    )

    fov_v = math.degrees(
        2 * math.atan(math.tan(math.radians(tactical_fov / 2)) * (out_h / out_w))
    )

    writer = FFmpegFrameWriter(
        output_path, fps,
        width=out_w, height=out_h,
        codec=codec, crf=crf, preset=preset,
    )

    try:
        for i, frame in enumerate(reader):
            perspective = py360convert.e2p(
                frame,
                fov_deg=(tactical_fov, fov_v),
                u_deg=tactical_yaw,
                v_deg=tactical_pitch,
                out_hw=(out_h, out_w),
                mode="bilinear",
            )
            if i >= skip_count:
                writer.write(perspective)
    finally:
        writer.close()


class Reframer:
    """Render broadcast and tactical videos from 360 source."""

    def __init__(self, config: dict):
        ref_cfg = config.get("reframer", {})
        exp_cfg = config.get("exporter", {})

        self.out_w, self.out_h = ref_cfg.get("output_resolution", [1920, 1080])

        # source_downscale: null means use native resolution (no downscale)
        ds = ref_cfg.get("source_downscale")
        self.source_downscale = tuple(ds) if ds else None

        self.num_workers = ref_cfg.get("num_workers", 12)

        self.tactical_fov = ref_cfg.get("tactical_fov", 120)
        self.tactical_yaw = ref_cfg.get("tactical_yaw", 0.0)
        self.tactical_pitch = ref_cfg.get("tactical_pitch", -5.0)
        self.overlap_sec = ref_cfg.get("overlap_sec", 0.5)

        self.codec = exp_cfg.get("codec", "libx264")
        self.crf = exp_cfg.get("crf", 18)
        self.preset = exp_cfg.get("preset", "medium")

        # NVENC support
        encoder = exp_cfg.get("encoder", "cpu")
        if encoder == "nvenc":
            self.codec = "h264_nvenc"

    def _resolve_source_dims(self, meta: VideoMeta) -> tuple[int, int] | None:
        """Resolve source dimensions. None means no rescaling (native res)."""
        if self.source_downscale:
            return self.source_downscale
        return None

    def render_broadcast(
        self,
        video_path: str | Path,
        meta: VideoMeta,
        camera_path_file: Path,
        output_path: Path,
    ):
        """Render broadcast follow video using parallel segment rendering."""
        camera_path = load_json(camera_path_file)
        total_frames = len(camera_path)
        video_str = str(video_path)

        source_dims = self._resolve_source_dims(meta)
        src_w = source_dims[0] if source_dims else meta.width
        src_h = source_dims[1] if source_dims else meta.height

        overlap_frames = int(self.overlap_sec * meta.fps)

        logger.info(
            "Rendering broadcast: %d frames, %d workers, source=%dx%d, output=%dx%d, overlap=%d frames",
            total_frames, self.num_workers, src_w, src_h, self.out_w, self.out_h,
            overlap_frames,
        )

        # Split into segments
        workers = min(self.num_workers, total_frames)
        chunk_size = total_frames // workers
        remainder = total_frames % workers

        segments: list[Path] = []
        futures = []

        with ProcessPoolExecutor(max_workers=workers) as pool:
            offset = 0
            for i in range(workers):
                n = chunk_size + (1 if i < remainder else 0)
                seg_path = output_path.parent / f"broadcast_seg_{i:03d}.mp4"
                segments.append(seg_path)

                cam_slice = camera_path[offset : offset + n]

                futures.append(pool.submit(
                    _render_segment,
                    video_str,
                    cam_slice,
                    offset,
                    meta.fps,
                    src_w if source_dims else None,
                    src_h if source_dims else None,
                    self.out_w,
                    self.out_h,
                    self.codec,
                    self.crf,
                    self.preset,
                    str(seg_path),
                    overlap_frames if i > 0 else 0,
                ))
                offset += n

            for i, f in enumerate(futures):
                f.result()
                logger.info("Broadcast segment %d/%d complete", i + 1, workers)

        logger.info("Concatenating %d broadcast segments", len(segments))
        concat_segments(segments, output_path)

        for seg in segments:
            seg.unlink(missing_ok=True)

        logger.info("Broadcast render complete: %s", output_path)

    def render_tactical(
        self,
        video_path: str | Path,
        meta: VideoMeta,
        output_path: Path,
    ):
        """Render fixed wide-angle tactical view.

        Uses parallel segments for speed even though camera is fixed,
        because the per-frame e2p computation is still CPU-bound.
        """
        total_frames = meta.total_frames
        video_str = str(video_path)

        source_dims = self._resolve_source_dims(meta)
        src_w = source_dims[0] if source_dims else None
        src_h = source_dims[1] if source_dims else None
        overlap_frames = int(self.overlap_sec * meta.fps)

        logger.info(
            "Rendering tactical: %d frames, FOV=%d, output=%dx%d, overlap=%d frames",
            total_frames, self.tactical_fov, self.out_w, self.out_h, overlap_frames,
        )

        workers = min(self.num_workers, total_frames)
        chunk_size = total_frames // workers
        remainder = total_frames % workers

        segments: list[Path] = []
        futures = []

        with ProcessPoolExecutor(max_workers=workers) as pool:
            offset = 0
            for i in range(workers):
                n = chunk_size + (1 if i < remainder else 0)
                seg_path = output_path.parent / f"tactical_seg_{i:03d}.mp4"
                segments.append(seg_path)

                futures.append(pool.submit(
                    _render_tactical_segment,
                    video_str,
                    offset,
                    n,
                    meta.fps,
                    src_w,
                    src_h,
                    self.out_w,
                    self.out_h,
                    self.tactical_yaw,
                    self.tactical_pitch,
                    self.tactical_fov,
                    self.codec,
                    self.crf,
                    self.preset,
                    str(seg_path),
                    overlap_frames if i > 0 else 0,
                ))
                offset += n

            for i, f in enumerate(futures):
                f.result()
                logger.info("Tactical segment %d/%d complete", i + 1, workers)

        logger.info("Concatenating %d tactical segments", len(segments))
        concat_segments(segments, output_path)

        for seg in segments:
            seg.unlink(missing_ok=True)

        logger.info("Tactical render complete: %s", output_path)
