"""Shared utilities: ffmpeg I/O, video probing, config loading, logging."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import yaml


logger = logging.getLogger("soccer360")


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    codec: str


def probe_video(path: str | Path) -> VideoMeta:
    """Extract metadata from a video file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    video_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    # Parse FPS from r_frame_rate (e.g. "30/1" or "30000/1001")
    fps_parts = video_stream["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    duration = float(info["format"].get("duration", 0))
    total_frames = int(round(fps * duration))
    codec = video_stream.get("codec_name", "unknown")

    return VideoMeta(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        total_frames=total_frames,
        codec=codec,
    )


# ---------------------------------------------------------------------------
# FFmpeg frame reader (streaming via pipe)
# ---------------------------------------------------------------------------

class FFmpegFrameReader:
    """Stream-read decoded frames from a video via ffmpeg stdout pipe.

    Yields numpy arrays of shape (height, width, 3) in RGB24 format.
    Frames are never written to disk.
    """

    def __init__(
        self,
        video_path: str | Path,
        output_width: int | None = None,
        output_height: int | None = None,
        start_frame: int = 0,
        num_frames: int | None = None,
        fps: float | None = None,
    ):
        self.video_path = str(video_path)
        self.output_width = output_width
        self.output_height = output_height
        self.start_frame = start_frame
        self.num_frames = num_frames
        self._fps = fps
        self._proc: subprocess.Popen | None = None

    def _resolve_dims(self) -> tuple[int, int]:
        if self.output_width and self.output_height:
            return self.output_width, self.output_height
        meta = probe_video(self.video_path)
        return meta.width, meta.height

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        w, h = self._resolve_dims()

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        # Seek if needed
        if self.start_frame > 0:
            fps = self._fps or probe_video(self.video_path).fps
            start_time = self.start_frame / fps
            cmd.extend(["-ss", f"{start_time:.4f}"])

        cmd.extend(["-i", self.video_path])

        if self.num_frames is not None:
            cmd.extend(["-frames:v", str(self.num_frames)])

        # Scale filter
        if self.output_width and self.output_height:
            cmd.extend(["-vf", f"scale={self.output_width}:{self.output_height}"])

        cmd.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])

        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=w * h * 3 * 4,  # buffer ~4 frames
        )

        frame_size = w * h * 3
        try:
            while True:
                raw = self._proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                yield frame
        finally:
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait()

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()


# ---------------------------------------------------------------------------
# FFmpeg frame writer (streaming via pipe)
# ---------------------------------------------------------------------------

class FFmpegFrameWriter:
    """Write frames to a video file via ffmpeg stdin pipe.

    Accepts numpy arrays of shape (height, width, 3) in RGB24 format.
    """

    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium",
        pix_fmt_out: str = "yuv420p",
    ):
        self.output_path = str(output_path)
        self.fps = fps
        self.width = width
        self.height = height

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", pix_fmt_out,
            self.output_path,
        ]

        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=width * height * 3 * 4,
        )

    def write(self, frame: np.ndarray):
        """Write a single RGB24 frame."""
        self._proc.stdin.write(frame.tobytes())

    def close(self):
        """Finish writing and wait for ffmpeg to complete."""
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()
        if self._proc.returncode != 0:
            stderr = self._proc.stderr.read().decode() if self._proc.stderr else ""
            raise RuntimeError(f"ffmpeg encode failed: {stderr}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Video segment concatenation
# ---------------------------------------------------------------------------

def concat_segments(segment_paths: list[Path], output_path: Path, copy: bool = True):
    """Concatenate video segments using ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for seg in segment_paths:
            f.write(f"file '{seg}'\n")
        list_path = f.name

    codec_args = ["-c", "copy"] if copy else ["-c:v", "libx264", "-crf", "18"]

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        *codec_args,
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        Path(list_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Frame extraction (for hard-frame export)
# ---------------------------------------------------------------------------

def extract_frame(video_path: str | Path, frame_idx: int, fps: float, output_path: Path):
    """Extract a single frame from a video using ffmpeg."""
    timestamp = frame_idx / fps
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{timestamp:.4f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Detection I/O (JSONL format)
# ---------------------------------------------------------------------------

def write_detections_jsonl(detections: list[dict], path: Path):
    """Write detections as JSON Lines (one detection per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for det in detections:
            f.write(json.dumps(det) + "\n")


def load_detections_jsonl(path: Path) -> list[dict]:
    """Load detections from a JSONL file."""
    detections = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                detections.append(json.loads(line))
    return detections


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def write_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Load pipeline YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Equirectangular angle helpers
# ---------------------------------------------------------------------------

def pixel_to_yaw_pitch(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    """Convert pixel position in equirectangular image to (yaw, pitch) degrees.

    Equirectangular mapping:
      x=0 -> yaw=-180, x=w -> yaw=+180
      y=0 -> pitch=+90 (top), y=h -> pitch=-90 (bottom)
    """
    yaw = (x / w) * 360.0 - 180.0
    pitch = 90.0 - (y / h) * 180.0
    return yaw, pitch


def wrap_angle_deg(a: float) -> float:
    """Wrap angle to (-180, 180] range.

    Consistent half-open interval: -180 maps to +180, 0 stays 0.
    """
    a = a % 360.0
    if a > 180.0:
        a -= 360.0
    return a


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def group_by_frame(detections: list[dict]) -> dict[int, list[dict]]:
    """Group detections list by frame index."""
    grouped: dict[int, list[dict]] = {}
    for det in detections:
        frame = det["frame"]
        grouped.setdefault(frame, []).append(det)
    return grouped


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configure structured logging for the pipeline."""
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, handlers=handlers)
