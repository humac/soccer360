"""Heuristic highlight detection and clip export.

Identifies interesting moments (shots, fast ball movement, goal-box entries,
player convergence, fast breaks) from detection, tracking, and player cluster
data, then exports ranked short video clips.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np

from .utils import VideoMeta, load_json

logger = logging.getLogger("soccer360.highlights")

# Event types that come from ball tracking vs cluster data
_BALL_EVENT_TYPES = {"speed", "goal_box", "direction_change"}
_CLUSTER_EVENT_TYPES = {
    "cluster_convergence",
    "cluster_velocity",
    "cluster_goal_zone",
    "cluster_density",
}
_CAMERA_EVENT_TYPES = {"camera_motion"}
_CONTEXT_EVENT_TYPES = {"goal_box", "cluster_convergence", "cluster_goal_zone"}
_MOTION_ONLY_EVENT_TYPES = {
    "speed",
    "direction_change",
    "cluster_velocity",
    "cluster_density",
    "camera_motion",
}


class HighlightDetector:
    """Detect highlights using ball movement and player cluster heuristics."""

    def __init__(self, config: dict):
        hl_cfg = config.get("highlights", {})

        # Existing ball-based config
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

        # Cluster-based detector config
        self.cluster_convergence_window = hl_cfg.get("cluster_convergence_window", 10)
        self.cluster_convergence_deg = hl_cfg.get("cluster_convergence_deg", 8.0)
        self.cluster_velocity_window = hl_cfg.get("cluster_velocity_window", 5)
        self.cluster_velocity_deg_per_sec = hl_cfg.get(
            "cluster_velocity_deg_per_sec", 15.0
        )
        self.cluster_goal_zone_regions = hl_cfg.get("cluster_goal_zone_regions", None)
        self.cluster_density_percentile = hl_cfg.get("cluster_density_percentile", 90)
        self.cluster_density_min_players = hl_cfg.get(
            "cluster_density_min_players",
            config.get("center_of_play", {}).get("min_players", 5),
        )
        self.camera_motion_window = hl_cfg.get("camera_motion_window", 5)
        self.camera_motion_deg_per_sec = hl_cfg.get("camera_motion_deg_per_sec", 12.0)
        self.camera_zoom_delta = hl_cfg.get("camera_zoom_delta", 4.0)
        self.same_type_cooldown_sec = hl_cfg.get("same_type_cooldown_sec", 0.75)
        self.motion_only_penalty = hl_cfg.get("motion_only_penalty", 0.8)

        # Scoring and ranking config
        self.score_weights = hl_cfg.get("score_weights", {
            "speed": 1.0,
            "goal_box": 1.5,
            "direction_change": 0.8,
            "cluster_convergence": 1.2,
            "cluster_velocity": 0.7,
            "cluster_goal_zone": 1.3,
            "cluster_density": 0.5,
            "camera_motion": 0.8,
        })
        self.combined_signal_bonus = hl_cfg.get("combined_signal_bonus", 1.5)
        self.min_clip_score = hl_cfg.get("min_clip_score", 2.0)
        self.max_clips = hl_cfg.get("max_clips", 20)

        exp_cfg = config.get("exporter", {})
        self.codec = exp_cfg.get("codec", "libx264")
        self.crf = exp_cfg.get("crf", 18)

        det_cfg = config.get("detector", {})
        self.det_w = det_cfg.get("resolution", [1920, 960])[0]
        self.det_h = det_cfg.get("resolution", [1920, 960])[1]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect_and_export(
        self,
        broadcast_path: Path,
        meta: VideoMeta,
        camera_path_file: Path,
        tracks_path: Path | None,
        output_dir: Path,
        player_cluster_path: Path | None = None,
    ):
        """Detect highlight events and export ranked clips."""
        fps = meta.fps
        events: list[dict] = []
        detector_stats: dict[str, int] = {}

        # Ball-based detectors (when tracks available)
        if tracks_path is not None and tracks_path.exists():
            tracks = load_json(tracks_path)
            velocities = self._compute_velocities(tracks, fps)

            speed_events = self._detect_speed_events(velocities, fps)
            goal_box_events = self._detect_goal_box_events(tracks, fps)
            direction_events = self._detect_direction_changes(velocities, fps)

            events.extend(speed_events)
            events.extend(goal_box_events)
            events.extend(direction_events)

            detector_stats["speed_events"] = len(speed_events)
            detector_stats["goal_box_events"] = len(goal_box_events)
            detector_stats["direction_change_events"] = len(direction_events)

        # Cluster-based detectors (when cluster data available)
        clusters = self._load_cluster_data(player_cluster_path)
        if clusters is not None:
            conv_events = self._detect_cluster_convergence(clusters, fps)
            vel_events = self._detect_cluster_velocity(clusters, fps)
            zone_events = self._detect_cluster_goal_zone(clusters, fps)
            density_events = self._detect_cluster_density_spike(clusters, fps)

            events.extend(conv_events)
            events.extend(vel_events)
            events.extend(zone_events)
            events.extend(density_events)

            detector_stats["cluster_convergence_events"] = len(conv_events)
            detector_stats["cluster_velocity_events"] = len(vel_events)
            detector_stats["cluster_goal_zone_events"] = len(zone_events)
            detector_stats["cluster_density_events"] = len(density_events)

        camera_entries = self._load_camera_path(camera_path_file)
        if camera_entries is not None:
            camera_motion_events = self._detect_camera_motion(camera_entries, fps)
            events.extend(camera_motion_events)
            detector_stats["camera_motion_events"] = len(camera_motion_events)

        detector_stats["total_raw_events"] = len(events)
        events = self._apply_same_type_cooldown(events)
        detector_stats["events_after_same_type_cooldown"] = len(events)
        detector_stats["ball_tracking_available"] = tracks_path is not None
        detector_stats["cluster_data_available"] = clusters is not None
        detector_stats["camera_path_available"] = camera_entries is not None

        if not events:
            logger.info("No highlight events detected")
            return

        logger.info(
            "Detected %d raw highlight events (%d after cooldown)",
            detector_stats["total_raw_events"],
            len(events),
        )

        # Cluster events into scored clips
        clips = self._cluster_events(events, fps)
        logger.info(
            "Clustered into %d highlight clips (score range: %.1f - %.1f)",
            len(clips),
            clips[-1]["score"] if clips else 0,
            clips[0]["score"] if clips else 0,
        )

        if not clips:
            logger.info("No clips above min_clip_score (%.1f)", self.min_clip_score)
            return

        # Export each clip from the broadcast video
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, clip in enumerate(clips):
            clip_path = output_dir / f"highlight_{i:03d}.mp4"
            self._export_clip(broadcast_path, clip, clip_path)
            logger.info(
                "Exported highlight %d (rank %d, score %.1f): %.1fs - %.1fs (%s)",
                i, clip["rank"], clip["score"],
                clip["start_sec"], clip["end_sec"],
                ", ".join(clip["event_types"]),
            )

        # Write manifest
        self._write_manifest(output_dir, clips, detector_stats)

    # ------------------------------------------------------------------
    # Cluster data loading
    # ------------------------------------------------------------------

    def _load_cluster_data(self, path: Path | None) -> list[dict] | None:
        """Load player cluster data if available."""
        if path is None:
            return None
        if not path.exists():
            return None
        try:
            return load_json(path)
        except Exception:
            logger.warning("Failed to load cluster data from %s", path)
            return None

    def _load_camera_path(self, path: Path | None) -> list[dict] | None:
        """Load camera path data if available."""
        if path is None:
            return None
        if not path.exists():
            return None
        try:
            return load_json(path)
        except Exception:
            logger.warning("Failed to load camera path from %s", path)
            return None

    # ------------------------------------------------------------------
    # Ball-based detectors (unchanged)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Cluster-based detectors (NEW)
    # ------------------------------------------------------------------

    def _detect_cluster_convergence(
        self, clusters: list[dict], fps: float
    ) -> list[dict]:
        """Detect rapid player convergence (set pieces, contested ball).

        Triggers when player spread decreases by more than threshold
        over a sliding window.
        """
        events = []
        window = self.cluster_convergence_window
        threshold = self.cluster_convergence_deg

        for i in range(window, len(clusters)):
            curr_c = clusters[i].get("cluster")
            prev_c = clusters[i - window].get("cluster")
            if curr_c is None or prev_c is None:
                continue

            decrease = prev_c["spread_x_deg"] - curr_c["spread_x_deg"]
            if decrease >= threshold:
                events.append({
                    "frame": clusters[i]["frame"],
                    "time_sec": clusters[i]["frame"] / fps,
                    "type": "cluster_convergence",
                    "value": decrease,
                })

        return events

    def _detect_cluster_velocity(
        self, clusters: list[dict], fps: float
    ) -> list[dict]:
        """Detect rapid cluster centroid movement (fast breaks, counter-attacks).

        Computes centroid displacement in degrees over a sliding window.
        """
        events = []
        window = self.cluster_velocity_window
        threshold = self.cluster_velocity_deg_per_sec

        for i in range(window, len(clusters)):
            curr_c = clusters[i].get("cluster")
            prev_c = clusters[i - window].get("cluster")
            if curr_c is None or prev_c is None:
                continue

            dx_px = curr_c["x"] - prev_c["x"]
            dy_px = curr_c["y"] - prev_c["y"]
            # Convert pixel displacement to degrees
            dx_deg = (dx_px / self.det_w) * 360.0
            dy_deg = (dy_px / self.det_h) * 180.0
            dist_deg = math.sqrt(dx_deg ** 2 + dy_deg ** 2)
            duration_sec = window / fps
            velocity_deg_per_sec = dist_deg / duration_sec if duration_sec > 0 else 0

            if velocity_deg_per_sec >= threshold:
                events.append({
                    "frame": clusters[i]["frame"],
                    "time_sec": clusters[i]["frame"] / fps,
                    "type": "cluster_velocity",
                    "value": velocity_deg_per_sec,
                })

        return events

    def _detect_cluster_goal_zone(
        self, clusters: list[dict], fps: float
    ) -> list[dict]:
        """Detect player cluster in goal zone (attacking pressure)."""
        events = []
        regions = self.cluster_goal_zone_regions or self.goal_box_regions

        for entry in clusters:
            c = entry.get("cluster")
            if c is None:
                continue
            # Need meaningful attacking presence
            if c["player_count"] < 6:
                continue

            nx = c["x"] / self.det_w
            ny = c["y"] / self.det_h

            for region in regions:
                x1, y1, x2, y2 = region
                if x1 <= nx <= x2 and y1 <= ny <= y2:
                    events.append({
                        "frame": entry["frame"],
                        "time_sec": entry["frame"] / fps,
                        "type": "cluster_goal_zone",
                        "value": float(c["player_count"]),
                    })
                    break

        return events

    def _detect_cluster_density_spike(
        self, clusters: list[dict], fps: float
    ) -> list[dict]:
        """Detect unusually high player count (set pieces, corners)."""
        counts = [
            entry["cluster"]["player_count"]
            for entry in clusters
            if entry.get("cluster") is not None
            and entry["cluster"].get("player_count", 0) >= self.cluster_density_min_players
            and entry["cluster"].get("confidence", 0.0) > 0.0
        ]
        if not counts:
            return []

        threshold = float(np.percentile(counts, self.cluster_density_percentile))

        events = []
        for entry in clusters:
            c = entry.get("cluster")
            if c is None:
                continue
            if c.get("player_count", 0) < self.cluster_density_min_players:
                continue
            if c.get("confidence", 0.0) <= 0.0:
                continue
            if c["player_count"] >= threshold:
                events.append({
                    "frame": entry["frame"],
                    "time_sec": entry["frame"] / fps,
                    "type": "cluster_density",
                    "value": float(c["player_count"]),
                })

        return events

    def _detect_camera_motion(
        self, camera_entries: list[dict], fps: float
    ) -> list[dict]:
        """Detect strong camera pan/zoom moments from the generated camera path."""
        events = []
        window = max(1, int(self.camera_motion_window))
        duration_sec = window / fps if fps > 0 else 0.0
        if duration_sec <= 0:
            return events

        for i in range(window, len(camera_entries)):
            curr = camera_entries[i]
            prev = camera_entries[i - window]
            if not curr or not prev:
                continue

            yaw_delta = abs(self._angle_delta_deg(curr.get("yaw", 0.0), prev.get("yaw", 0.0)))
            pitch_delta = abs(curr.get("pitch", 0.0) - prev.get("pitch", 0.0))
            fov_delta = abs(curr.get("fov", 0.0) - prev.get("fov", 0.0))
            pan_distance_deg = math.hypot(yaw_delta, pitch_delta)
            pan_speed_deg_per_sec = pan_distance_deg / duration_sec

            if (
                pan_speed_deg_per_sec < self.camera_motion_deg_per_sec
                and fov_delta < self.camera_zoom_delta
            ):
                continue

            value = max(
                pan_speed_deg_per_sec / max(self.camera_motion_deg_per_sec, 1e-6),
                fov_delta / max(self.camera_zoom_delta, 1e-6),
            )
            events.append({
                "frame": i,
                "time_sec": i / fps,
                "type": "camera_motion",
                "value": value,
                "pan_speed_deg_per_sec": pan_speed_deg_per_sec,
                "fov_delta": fov_delta,
            })

        return events

    # ------------------------------------------------------------------
    # Clustering, scoring, and export
    # ------------------------------------------------------------------

    def _cluster_events(self, events: list[dict], fps: float) -> list[dict]:
        """Cluster nearby events into scored, ranked highlight clips."""
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

        # Convert clusters to scored clips with margins
        clips = []
        for cluster in clusters:
            start_sec = max(0, cluster[0]["time_sec"] - self.pre_margin_sec)
            end_sec = cluster[-1]["time_sec"] + self.post_margin_sec

            duration = end_sec - start_sec
            if duration < self.min_clip_duration_sec:
                continue

            event_types = sorted(set(e["type"] for e in cluster))

            # Compute score
            total_score = sum(
                self._score_event(e) for e in cluster
            )

            # Combined signal bonus: clip has multiple signal families.
            has_ball = any(e["type"] in _BALL_EVENT_TYPES for e in cluster)
            has_cluster = any(e["type"] in _CLUSTER_EVENT_TYPES for e in cluster)
            has_camera = any(e["type"] in _CAMERA_EVENT_TYPES for e in cluster)
            if sum((has_ball, has_cluster, has_camera)) >= 2:
                total_score *= self.combined_signal_bonus

            # Down-rank generic motion-only clips to reduce midfield churn.
            if set(event_types).issubset(_MOTION_ONLY_EVENT_TYPES):
                total_score *= self.motion_only_penalty

            if total_score < self.min_clip_score:
                continue

            clips.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration": duration,
                "event_count": len(cluster),
                "event_types": event_types,
                "score": round(total_score, 2),
                "rank": 0,  # filled below
            })

        # Sort by score descending, cap at max_clips
        clips.sort(key=lambda c: c["score"], reverse=True)
        clips = clips[: self.max_clips]

        # Assign rank, then re-sort by time for sequential export
        for i, clip in enumerate(clips):
            clip["rank"] = i + 1
        clips.sort(key=lambda c: c["start_sec"])

        return clips

    def _apply_same_type_cooldown(self, events: list[dict]) -> list[dict]:
        """Collapse same-type bursts into their strongest nearby event."""
        if not events:
            return []
        if self.same_type_cooldown_sec <= 0:
            return sorted(events, key=lambda e: e["time_sec"])

        kept_by_type: dict[str, list[dict]] = {}
        for event in sorted(events, key=lambda e: e["time_sec"]):
            event_type = event["type"]
            same_type_events = kept_by_type.setdefault(event_type, [])
            if not same_type_events:
                same_type_events.append(event.copy())
                continue

            prev = same_type_events[-1]
            if event["time_sec"] - prev["time_sec"] > self.same_type_cooldown_sec:
                same_type_events.append(event.copy())
                continue

            if self._event_priority(event) >= self._event_priority(prev):
                same_type_events[-1] = event.copy()

        collapsed = [
            event
            for events_for_type in kept_by_type.values()
            for event in events_for_type
        ]
        collapsed.sort(key=lambda e: e["time_sec"])
        return collapsed

    def _score_event(self, event: dict) -> float:
        """Base score contribution for a single event."""
        return float(self.score_weights.get(event["type"], 1.0))

    def _event_priority(self, event: dict) -> float:
        """Prefer the strongest event when collapsing bursts."""
        return float(event.get("value", 0.0))

    def _angle_delta_deg(self, current: float, previous: float) -> float:
        """Shortest signed angular delta in degrees."""
        delta = (current - previous) % 360.0
        if delta > 180.0:
            delta -= 360.0
        return delta

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

    def _write_manifest(
        self,
        output_dir: Path,
        clips: list[dict],
        detector_stats: dict,
    ):
        """Write highlights.json manifest with clip metadata and stats."""
        manifest = {
            "clip_count": len(clips),
            "clips": [
                {
                    "filename": f"highlight_{i:03d}.mp4",
                    "start_sec": clip["start_sec"],
                    "end_sec": clip["end_sec"],
                    "duration": clip["duration"],
                    "score": clip["score"],
                    "rank": clip["rank"],
                    "event_types": clip["event_types"],
                    "event_count": clip["event_count"],
                }
                for i, clip in enumerate(clips)
            ],
            "detector_stats": detector_stats,
        }
        manifest_path = output_dir / "highlights.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Wrote highlight manifest: %s", manifest_path)
