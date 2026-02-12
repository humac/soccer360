"""Automatic hard-frame export for active learning.

Analyzes detection and tracking results to identify frames where the model
struggled, exports those frames as images, and writes a structured manifest
for downstream labeling workflows.

Hard-frame criteria:
  1. Low detection confidence (below threshold)
  2. Consecutive frames with no ball detected (gap > N frames)
  3. Large position jumps between tracked frames (tracker instability)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from .utils import (
    VideoMeta,
    extract_frame,
    group_by_frame,
    load_detections_jsonl,
    load_json,
    write_json,
)

logger = logging.getLogger("soccer360.hard_frames")


class HardFrameExporter:
    """Identify and export hard frames during pipeline processing."""

    def __init__(self, config: dict):
        al_cfg = config.get("active_learning", {})
        self.enabled = al_cfg.get("enabled", True)
        self.confidence_threshold = al_cfg.get("confidence_threshold", 0.3)
        self.gap_frames = al_cfg.get("gap_frames", 15)
        self.max_export_frames = al_cfg.get("max_export_frames", 500)
        self.position_jump_px = al_cfg.get("position_jump_px", 150)

        self.labeling_base = Path(
            config.get("paths", {}).get("labeling", "/tank/labeling")
        )

    def run(
        self,
        video_path: str,
        meta: VideoMeta,
        detections_path: Path,
        tracks_path: Path,
        work_dir: Path,
    ):
        """Analyze detections + tracks and export hard frames.

        Writes frames as JPEG to /tank/labeling/{match_name}/frames/ and a
        manifest to /tank/labeling/{match_name}/hard_frames.json.
        Also copies the manifest to work_dir for artifact preservation.
        """
        if not self.enabled:
            logger.info("Hard frame export disabled")
            return

        detections = load_detections_jsonl(detections_path)
        tracks = load_json(tracks_path)

        match_name = Path(video_path).stem

        hard_frames = self._identify_hard_frames(detections, tracks, meta)

        if not hard_frames:
            logger.info("No hard frames identified for %s", match_name)
            return

        logger.info("Identified %d hard frame candidates for %s", len(hard_frames), match_name)

        # Sample if too many
        if len(hard_frames) > self.max_export_frames:
            hard_frames = random.sample(hard_frames, self.max_export_frames)
            hard_frames.sort(key=lambda h: h["frame_index"])
            logger.info("Sampled down to %d frames", self.max_export_frames)

        # Export frames
        output_dir = self.labeling_base / match_name
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        exported = []
        for hf in hard_frames:
            frame_idx = hf["frame_index"]
            out_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            try:
                extract_frame(video_path, frame_idx, meta.fps, out_path)
                hf["exported_path"] = str(out_path)
                exported.append(hf)
            except Exception:
                logger.warning("Failed to extract frame %d", frame_idx)

        # Write manifest
        manifest = {
            "video": str(video_path),
            "match_name": match_name,
            "total_frames_analyzed": len(tracks),
            "hard_frames_identified": len(hard_frames),
            "hard_frames_exported": len(exported),
            "criteria": {
                "confidence_threshold": self.confidence_threshold,
                "gap_frames": self.gap_frames,
                "position_jump_px": self.position_jump_px,
            },
            "frames": exported,
        }
        manifest_path = output_dir / "hard_frames.json"
        write_json(manifest, manifest_path)

        # Copy manifest to work_dir for inclusion in pipeline artifacts
        write_json(manifest, work_dir / "hard_frames.json")

        logger.info(
            "Exported %d hard frames to %s (manifest: %s)",
            len(exported), frames_dir, manifest_path,
        )

    def _identify_hard_frames(
        self,
        detections: list[dict],
        tracks: list[dict],
        meta: VideoMeta,
    ) -> list[dict]:
        """Identify frames where detection/tracking quality was poor.

        Returns a sorted list of hard frame entries with frame_index,
        timestamp_sec, reasons, and optional predicted bbox/confidence.
        """
        hard_frames: dict[int, dict] = {}

        # Build per-frame detection lookup
        by_frame = group_by_frame(detections)

        # Criterion 1: Low confidence detections
        for det in detections:
            if det["confidence"] < self.confidence_threshold:
                idx = det["frame"]
                entry = hard_frames.setdefault(idx, {
                    "frame_index": idx,
                    "timestamp_sec": round(idx / meta.fps, 3),
                    "reasons": [],
                })
                if "low_confidence" not in entry["reasons"]:
                    entry["reasons"].append("low_confidence")
                    entry["predicted_bbox"] = det["bbox"]
                    entry["predicted_confidence"] = det["confidence"]

        # Criterion 2: Consecutive lost-ball gaps exceeding threshold
        consecutive_lost = 0
        gap_start = -1
        for i, track in enumerate(tracks):
            if track.get("ball") is None:
                if consecutive_lost == 0:
                    gap_start = i
                consecutive_lost += 1
            else:
                if consecutive_lost >= self.gap_frames:
                    for f in range(gap_start, gap_start + consecutive_lost):
                        entry = hard_frames.setdefault(f, {
                            "frame_index": f,
                            "timestamp_sec": round(f / meta.fps, 3),
                            "reasons": [],
                        })
                        if "lost_ball_gap" not in entry["reasons"]:
                            entry["reasons"].append("lost_ball_gap")
                            entry["gap_length"] = consecutive_lost
                consecutive_lost = 0

        # Handle trailing gap
        if consecutive_lost >= self.gap_frames:
            for f in range(gap_start, gap_start + consecutive_lost):
                entry = hard_frames.setdefault(f, {
                    "frame_index": f,
                    "timestamp_sec": round(f / meta.fps, 3),
                    "reasons": [],
                })
                if "lost_ball_gap" not in entry["reasons"]:
                    entry["reasons"].append("lost_ball_gap")
                    entry["gap_length"] = consecutive_lost

        # Criterion 3: Large position jumps between consecutive ball positions
        prev_ball = None
        for i, track in enumerate(tracks):
            ball = track.get("ball")
            if ball is not None:
                if prev_ball is not None:
                    dx = ball["x"] - prev_ball["x"]
                    dy = ball["y"] - prev_ball["y"]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist > self.position_jump_px:
                        entry = hard_frames.setdefault(i, {
                            "frame_index": i,
                            "timestamp_sec": round(i / meta.fps, 3),
                            "reasons": [],
                        })
                        if "position_jump" not in entry["reasons"]:
                            entry["reasons"].append("position_jump")
                            entry["jump_distance_px"] = round(dist, 1)
                prev_ball = ball
            else:
                prev_ball = None

        return sorted(hard_frames.values(), key=lambda h: h["frame_index"])
