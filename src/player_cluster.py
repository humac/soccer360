"""Center-of-play estimation from player cluster positions.

Computes a per-frame centroid of detected players as a fallback/blend
signal for camera tracking when ball detection is unreliable.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from .utils import load_detections_jsonl, pixel_to_yaw_pitch, write_json

logger = logging.getLogger(__name__)


class PlayerClusterComputer:
    """Compute per-frame player cluster centroid and spread from detections."""

    def __init__(self, config: dict):
        cop_cfg = config.get("center_of_play", {})
        self.player_class = cop_cfg.get("player_class", 0)
        self.min_player_conf = cop_cfg.get("min_player_conf", 0.30)
        self.trim_fraction = cop_cfg.get("trim_fraction", 0.10)
        self.min_players = cop_cfg.get("min_players", 4)
        self.ema_alpha = cop_cfg.get("ema_alpha", 0.20)

        # Detection resolution for pixel→angle conversion
        det_cfg = config.get("detection", {})
        img_size = det_cfg.get("img_size", 960)
        self.det_width = img_size * 2
        self.det_height = img_size

    def run(
        self,
        detections_path: Path,
        output_path: Path,
        total_frames: int,
    ) -> None:
        """Read detections, filter to persons, compute per-frame cluster."""
        detections = load_detections_jsonl(detections_path)

        # Filter to player detections only
        player_dets = [
            d for d in detections
            if d.get("class_id", d.get("class", -1)) == self.player_class
            and d.get("conf", d.get("confidence", 0.0)) >= self.min_player_conf
        ]

        # Group by frame
        by_frame: dict[int, list[dict]] = {}
        for det in player_dets:
            frame = det.get("frame_index", det.get("frame", -1))
            by_frame.setdefault(frame, []).append(det)

        logger.info(
            "Player cluster: %d player detections across %d frames (of %d total)",
            len(player_dets), len(by_frame), total_frames,
        )

        # Compute raw per-frame clusters
        raw_clusters: list[dict | None] = []
        for frame_idx in range(total_frames):
            dets = by_frame.get(frame_idx)
            if dets is None or len(dets) < self.min_players:
                raw_clusters.append(None)
                continue
            raw_clusters.append(self._compute_cluster(dets))

        # EMA temporal smoothing
        smoothed = self._ema_smooth(raw_clusters)

        # Build output
        result = []
        for frame_idx in range(total_frames):
            result.append({
                "frame": frame_idx,
                "cluster": smoothed[frame_idx],
            })

        write_json(result, output_path)

        valid_count = sum(1 for c in smoothed if c is not None)
        logger.info(
            "Player cluster output: %d/%d frames with valid cluster",
            valid_count, total_frames,
        )

    def _compute_cluster(self, dets: list[dict]) -> dict:
        """Compute trimmed-mean centroid and spread from player detections."""
        # Extract bbox centroids
        centroids = []
        confs = []
        for det in dets:
            bbox = det.get("bbox_xyxy", det.get("bbox", [0, 0, 0, 0]))
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            centroids.append((cx, cy))
            confs.append(det.get("conf", det.get("confidence", 0.0)))

        # Sort by x for trimmed mean
        sorted_by_x = sorted(centroids, key=lambda p: p[0])
        n = len(sorted_by_x)
        trim_count = max(1, int(n * self.trim_fraction))

        # Discard outliers from each end (isolated GKs, etc.)
        if n > trim_count * 2 + self.min_players:
            trimmed = sorted_by_x[trim_count: n - trim_count]
        else:
            trimmed = sorted_by_x

        # Compute centroid
        mean_x = sum(p[0] for p in trimmed) / len(trimmed)
        mean_y = sum(p[1] for p in trimmed) / len(trimmed)

        # Compute spread in degrees (std dev of x positions, converted to angle)
        if len(trimmed) > 1:
            x_vals = [p[0] for p in trimmed]
            x_mean = mean_x
            variance = sum((x - x_mean) ** 2 for x in x_vals) / len(x_vals)
            std_x_px = math.sqrt(variance)
            # Convert pixel spread to approximate degree spread
            # In equirectangular: 1 pixel = 360/width degrees horizontally
            spread_x_deg = std_x_px * (360.0 / self.det_width)
        else:
            spread_x_deg = 0.0

        mean_conf = sum(confs) / len(confs) if confs else 0.0

        return {
            "x": round(mean_x, 1),
            "y": round(mean_y, 1),
            "spread_x_deg": round(spread_x_deg, 1),
            "player_count": len(dets),
            "confidence": round(mean_conf, 3),
        }

    def _ema_smooth(
        self, raw_clusters: list[dict | None]
    ) -> list[dict | None]:
        """Apply EMA temporal smoothing to cluster centroids."""
        alpha = self.ema_alpha
        smoothed: list[dict | None] = []
        prev_x: float | None = None
        prev_y: float | None = None

        for cluster in raw_clusters:
            if cluster is None:
                # Carry forward previous smooth value if available
                if prev_x is not None:
                    smoothed.append({
                        "x": round(prev_x, 1),
                        "y": round(prev_y, 1),
                        "spread_x_deg": smoothed[-1]["spread_x_deg"] if smoothed and smoothed[-1] else 0.0,
                        "player_count": 0,
                        "confidence": 0.0,
                    })
                else:
                    smoothed.append(None)
                continue

            cx, cy = cluster["x"], cluster["y"]
            if prev_x is None:
                prev_x = cx
                prev_y = cy
            else:
                prev_x = alpha * cx + (1 - alpha) * prev_x
                prev_y = alpha * cy + (1 - alpha) * prev_y

            smoothed.append({
                "x": round(prev_x, 1),
                "y": round(prev_y, 1),
                "spread_x_deg": cluster["spread_x_deg"],
                "player_count": cluster["player_count"],
                "confidence": cluster["confidence"],
            })

        return smoothed
