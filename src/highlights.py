"""Heuristic highlight detection and clip export.

Identifies interesting moments (shots, fast ball movement, goal-box entries)
from detection and tracking data, then exports short video clips.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

import numpy as np

from .utils import VideoMeta, load_json

logger = logging.getLogger("soccer360.highlights")


class HighlightDetector:
    """Detect highlights using ball movement heuristics and export clips."""

    def __init__(self, config: dict):
        hl_cfg = config.get("highlights", {})

        self.speed_percentile = hl_cfg.get("speed_percentile", 95)
        self.direction_change_deg = hl_cfg.get("direction_change_deg", 90)
        self.goal_box_regions = hl_cfg.get("goal_box_regions", [
            [0.0, 0.3, 0.08, 0.7],
            [0.92, 0.3, 1.0, 0.7],
        ])
        self.pre_margin_sec = hl_cfg.get("pre_margin_sec", 5.0)
        self.post_margin_sec = hl_cfg.get("post_margin_sec", 3.0)
        self.min_clip_gap_sec = hl_cfg.get("min_clip_gap_sec", 5.0)
        self.min_clip_duration_sec = hl_cfg.get("min_clip_duration_sec", 3.0)

        exp_cfg = config.get("exporter", {})
        self.codec = exp_cfg.get("codec", "libx264")
        self.crf = exp_cfg.get("crf", 18)

        det_cfg = config.get("detector", {})
        self.det_w = det_cfg.get("resolution", [1920, 960])[0]
        self.det_h = det_cfg.get("resolution", [1920, 960])[1]

    def detect_and_export(
        self,
        broadcast_path: Path,
        meta: VideoMeta,
        camera_path_file: Path,
        tracks_path: Path,
        output_dir: Path,
    ):
        """Detect highlight events and export clips from broadcast video."""
        tracks = load_json(tracks_path)
        camera_path = load_json(camera_path_file)
        fps = meta.fps

        # Compute per-frame ball velocities (in detection pixel space)
        velocities = self._compute_velocities(tracks, fps)

        # Detect events
        events: list[dict] = []
        events.extend(self._detect_speed_events(velocities, fps))
        events.extend(self._detect_goal_box_events(tracks, fps))
        events.extend(self._detect_direction_changes(velocities, fps))

        if not events:
            logger.info("No highlight events detected")
            return

        logger.info("Detected %d raw highlight events", len(events))

        # Cluster events into clips
        clips = self._cluster_events(events, fps)
        logger.info("Clustered into %d highlight clips", len(clips))

        # Export each clip from the broadcast video
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, clip in enumerate(clips):
            clip_path = output_dir / f"highlight_{i:03d}.mp4"
            self._export_clip(broadcast_path, clip, clip_path)
            logger.info(
                "Exported highlight %d: %.1fs - %.1fs (%s)",
                i, clip["start_sec"], clip["end_sec"],
                ", ".join(clip["event_types"]),
            )

    def _compute_velocities(
        self, tracks: list[dict], fps: float
    ) -> list[dict]:
        """Compute ball velocity and acceleration per frame."""
        velocities = []

        for i, t in enumerate(tracks):
            entry = {"frame": i, "speed": 0.0, "vx": 0.0, "vy": 0.0, "has_ball": False}

            if t.get("ball") is not None and i > 0 and tracks[i - 1].get("ball") is not None:
                curr = t["ball"]
                prev = tracks[i - 1]["ball"]
                dx = curr["x"] - prev["x"]
                dy = curr["y"] - prev["y"]
                # Pixels per second
                vx = dx * fps
                vy = dy * fps
                speed = math.sqrt(vx ** 2 + vy ** 2)
                entry.update({"speed": speed, "vx": vx, "vy": vy, "has_ball": True})
            elif t.get("ball") is not None:
                entry["has_ball"] = True

            velocities.append(entry)

        return velocities

    def _detect_speed_events(
        self, velocities: list[dict], fps: float
    ) -> list[dict]:
        """Detect frames where ball speed exceeds threshold."""
        speeds = [v["speed"] for v in velocities if v["speed"] > 0]
        if not speeds:
            return []

        threshold = float(np.percentile(speeds, self.speed_percentile))
        logger.info("Speed event threshold: %.1f px/sec (p%d)", threshold, self.speed_percentile)

        events = []
        for v in velocities:
            if v["speed"] >= threshold:
                events.append({
                    "frame": v["frame"],
                    "time_sec": v["frame"] / fps,
                    "type": "speed",
                    "value": v["speed"],
                })
        return events

    def _detect_goal_box_events(
        self, tracks: list[dict], fps: float
    ) -> list[dict]:
        """Detect when ball enters goal-box regions."""
        events = []

        for t in tracks:
            if t.get("ball") is None:
                continue

            # Normalize ball position to [0, 1]
            nx = t["ball"]["x"] / self.det_w
            ny = t["ball"]["y"] / self.det_h

            for region in self.goal_box_regions:
                x1, y1, x2, y2 = region
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    events.append({
                        "frame": t["frame"],
                        "time_sec": t["frame"] / fps,
                        "type": "goal_box",
                        "value": 1.0,
                    })
                    break  # Only one event per frame

        return events

    def _detect_direction_changes(
        self, velocities: list[dict], fps: float
    ) -> list[dict]:
        """Detect sudden direction changes (kicks, headers, deflections)."""
        events = []
        threshold_rad = math.radians(self.direction_change_deg)

        for i in range(1, len(velocities)):
            curr = velocities[i]
            prev = velocities[i - 1]

            # Need significant velocity in both frames
            if curr["speed"] < 10 or prev["speed"] < 10:
                continue

            # Compute angle between velocity vectors
            dot = prev["vx"] * curr["vx"] + prev["vy"] * curr["vy"]
            mag = prev["speed"] * curr["speed"]
            if mag < 1e-6:
                continue

            cos_angle = np.clip(dot / mag, -1.0, 1.0)
            angle = math.acos(cos_angle)

            if angle >= threshold_rad:
                events.append({
                    "frame": curr["frame"],
                    "time_sec": curr["frame"] / fps,
                    "type": "direction_change",
                    "value": math.degrees(angle),
                })

        return events

    def _cluster_events(self, events: list[dict], fps: float) -> list[dict]:
        """Cluster nearby events into highlight clips with margins."""
        if not events:
            return []

        # Sort by time
        events.sort(key=lambda e: e["time_sec"])

        # Merge events within min_clip_gap_sec
        clusters: list[list[dict]] = [[events[0]]]
        for e in events[1:]:
            if e["time_sec"] - clusters[-1][-1]["time_sec"] <= self.min_clip_gap_sec:
                clusters[-1].append(e)
            else:
                clusters.append([e])

        # Convert clusters to clips with margins
        clips = []
        for cluster in clusters:
            start_sec = max(0, cluster[0]["time_sec"] - self.pre_margin_sec)
            end_sec = cluster[-1]["time_sec"] + self.post_margin_sec

            duration = end_sec - start_sec
            if duration < self.min_clip_duration_sec:
                continue

            event_types = list(set(e["type"] for e in cluster))
            clips.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration": duration,
                "event_count": len(cluster),
                "event_types": event_types,
            })

        return clips

    def _export_clip(self, source_video: Path, clip: dict, output_path: Path):
        """Extract a clip from the broadcast video using ffmpeg."""
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{clip['start_sec']:.3f}",
            "-i", str(source_video),
            "-t", f"{clip['duration']:.3f}",
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-c:a", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
