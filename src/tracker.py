"""ByteTrack-based ball tracking over detection results.

Minimal standalone ByteTrack implementation for single-object (ball) tracking.
Decoupled from YOLO -- operates on pre-computed detections.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils import group_by_frame, load_detections_jsonl, write_json

logger = logging.getLogger("soccer360.tracker")


# ---------------------------------------------------------------------------
# Kalman filter for bounding box tracking
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """Per-track Kalman filter operating on (cx, cy, area, aspect_ratio)."""

    _count = 0

    def __init__(self, bbox: np.ndarray, det_conf: float = 0.0):
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition: constant velocity model
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i + 4] = 1.0

        # Measurement matrix
        self.kf.H = np.eye(4, 8)

        # Covariance
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.R *= 1.0
        self.kf.Q[4:, 4:] *= 0.01

        # Init state from bbox
        z = self._bbox_to_z(bbox)
        self.kf.x[:4] = z.reshape(4, 1)

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_det_conf = det_conf

        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1,y1,x2,y2] to [cx, cy, area, aspect_ratio]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2
        cy = bbox[1] + h / 2
        area = w * h
        ar = w / (h + 1e-6)
        return np.array([cx, cy, area, ar])

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, area, aspect_ratio] to [x1,y1,x2,y2]."""
        cx, cy, area, ar = z.flatten()[:4]
        w = np.sqrt(max(area * ar, 1e-6))
        h = area / (w + 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> np.ndarray:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._z_to_bbox(self.kf.x[:4])

    def update(self, bbox: np.ndarray, det_conf: float = 0.0):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_det_conf = det_conf
        z = self._bbox_to_z(bbox)
        self.kf.update(z.reshape(4, 1))

    def get_state(self) -> np.ndarray:
        return self._z_to_bbox(self.kf.x[:4])


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def iou_batch(bb_a: np.ndarray, bb_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bboxes. Returns (len(a), len(b)) matrix."""
    xx1 = np.maximum(bb_a[:, 0:1], bb_b[:, 0].T)
    yy1 = np.maximum(bb_a[:, 1:2], bb_b[:, 1].T)
    xx2 = np.minimum(bb_a[:, 2:3], bb_b[:, 2].T)
    yy2 = np.minimum(bb_a[:, 3:4], bb_b[:, 3].T)

    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = (bb_a[:, 2] - bb_a[:, 0]) * (bb_a[:, 3] - bb_a[:, 1])
    area_b = (bb_b[:, 2] - bb_b[:, 0]) * (bb_b[:, 3] - bb_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# ByteTrack core
# ---------------------------------------------------------------------------

class ByteTrackInstance:
    """Minimal ByteTrack implementation.

    Two-stage association:
    1. Match high-confidence detections to existing tracks.
    2. Match remaining low-confidence detections to unmatched tracks.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.4,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh

        self.tracked: list[KalmanBoxTracker] = []
        self.lost: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, det_array: np.ndarray) -> list[dict]:
        """Update tracks with new detections.

        Args:
            det_array: (N, 5) array of [x1, y1, x2, y2, confidence].

        Returns:
            List of active track dicts with keys: track_id, bbox, score.
        """
        self.frame_count += 1

        # Predict existing tracks
        for trk in self.tracked + self.lost:
            trk.predict()

        # Split detections by confidence
        if len(det_array) > 0:
            high_mask = det_array[:, 4] >= self.track_high_thresh
            low_mask = (det_array[:, 4] >= self.track_low_thresh) & (~high_mask)
            high_dets = det_array[high_mask]
            low_dets = det_array[low_mask]
        else:
            high_dets = np.empty((0, 5))
            low_dets = np.empty((0, 5))

        # --- First association: high-conf dets vs tracked ---
        active_tracks = self.tracked + self.lost
        unmatched_trks_idx: list[int] = list(range(len(active_tracks)))
        unmatched_dets_idx: list[int] = list(range(len(high_dets)))
        matched_pairs: list[tuple[int, int]] = []

        if len(active_tracks) > 0 and len(high_dets) > 0:
            trk_boxes = np.array([t.get_state() for t in active_tracks])
            iou_matrix = iou_batch(trk_boxes, high_dets[:, :4])
            cost = 1.0 - iou_matrix

            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if iou_matrix[r, c] >= self.match_thresh:
                    matched_pairs.append((r, c))

            matched_trk = {r for r, _ in matched_pairs}
            matched_det = {c for _, c in matched_pairs}
            unmatched_trks_idx = [i for i in range(len(active_tracks)) if i not in matched_trk]
            unmatched_dets_idx = [i for i in range(len(high_dets)) if i not in matched_det]

        # Update matched tracks with detection confidence
        for trk_i, det_i in matched_pairs:
            active_tracks[trk_i].update(
                high_dets[det_i, :4], det_conf=float(high_dets[det_i, 4])
            )

        # --- Second association: low-conf dets vs unmatched tracked ---
        if len(low_dets) > 0 and unmatched_trks_idx:
            remaining_trks = [active_tracks[i] for i in unmatched_trks_idx]
            trk_boxes = np.array([t.get_state() for t in remaining_trks])
            iou_matrix = iou_batch(trk_boxes, low_dets[:, :4])
            cost = 1.0 - iou_matrix

            row_idx, col_idx = linear_sum_assignment(cost)
            matched_second = set()
            for r, c in zip(row_idx, col_idx):
                if iou_matrix[r, c] >= self.match_thresh:
                    remaining_trks[r].update(
                        low_dets[c, :4], det_conf=float(low_dets[c, 4])
                    )
                    matched_second.add(unmatched_trks_idx[r])

            unmatched_trks_idx = [i for i in unmatched_trks_idx if i not in matched_second]

        # --- Create new tracks from unmatched high-conf detections ---
        for det_i in unmatched_dets_idx:
            if high_dets[det_i, 4] >= self.new_track_thresh:
                new_trk = KalmanBoxTracker(
                    high_dets[det_i, :4], det_conf=float(high_dets[det_i, 4])
                )
                active_tracks.append(new_trk)

        # --- Manage lost/removed tracks ---
        new_tracked = []
        new_lost = []

        for i, trk in enumerate(active_tracks):
            if trk.time_since_update == 0:
                new_tracked.append(trk)
            elif trk.time_since_update <= self.track_buffer:
                new_lost.append(trk)
            # else: track removed (exceeded buffer)

        self.tracked = new_tracked
        self.lost = new_lost

        # Return active tracks with real detection confidence as score
        output = []
        for trk in self.tracked:
            bbox = trk.get_state()
            # Blend last detection confidence with hit-streak reliability
            streak_factor = min(trk.hit_streak / 5.0, 1.0)
            score = 0.7 * trk.last_det_conf + 0.3 * streak_factor
            output.append({
                "track_id": trk.id,
                "bbox": bbox.tolist(),
                "score": score,
            })
        return output


# ---------------------------------------------------------------------------
# High-level tracker interface
# ---------------------------------------------------------------------------

class Tracker:
    """Run ByteTrack over pre-computed detections to produce ball tracks.

    All pixel-space thresholds (max_speed_px, max_displacement_px, bbox area
    limits) are in detector.resolution coordinate space.  Detections arriving
    from the detector are already in that space, so no rescaling is needed.
    """

    def __init__(self, config: dict):
        trk_cfg = config.get("tracker", {})
        self.track_high_thresh = trk_cfg.get("track_high_thresh", 0.25)
        self.track_low_thresh = trk_cfg.get("track_low_thresh", 0.1)
        self.new_track_thresh = trk_cfg.get("new_track_thresh", 0.25)
        self.track_buffer = trk_cfg.get("track_buffer", 30)
        self.match_thresh = trk_cfg.get("match_thresh", 0.4)

        # Ball selection sanity checks (detector.resolution pixel space)
        self.max_speed_px = trk_cfg.get("max_speed_px_per_frame", 200)
        self.max_displacement_px = trk_cfg.get("max_displacement_px", 300)
        self.min_bbox_area = trk_cfg.get("min_bbox_area", 10)
        self.max_bbox_area = trk_cfg.get("max_bbox_area", 10000)

    def run(self, detections_path: Path, output_path: Path):
        """Process detections file and produce per-frame ball positions."""
        detections = load_detections_jsonl(detections_path)
        logger.info("Loaded %d detections from %s", len(detections), detections_path)

        bt = ByteTrackInstance(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
        )

        # Determine total frames
        if detections:
            max_frame = max(d["frame"] for d in detections)
        else:
            logger.warning("No detections found")
            write_json([], output_path)
            return

        by_frame = group_by_frame(detections)
        tracks: list[dict] = []
        prev_ball: dict | None = None

        for frame_idx in range(max_frame + 1):
            frame_dets = by_frame.get(frame_idx, [])

            if frame_dets:
                det_array = np.array(
                    [[*d["bbox"], d["confidence"]] for d in frame_dets],
                    dtype=np.float32,
                )
            else:
                det_array = np.empty((0, 5), dtype=np.float32)

            active = bt.update(det_array)
            ball = self._select_ball(active, frame_idx, prev_ball)
            tracks.append(ball)

            if ball["ball"] is not None:
                prev_ball = ball["ball"]

        logger.info(
            "Tracking complete: %d frames, ball found in %d",
            len(tracks),
            sum(1 for t in tracks if t["ball"] is not None),
        )
        write_json(tracks, output_path)

    def _select_ball(
        self, active_tracks: list[dict], frame_idx: int, prev_ball: dict | None
    ) -> dict:
        """Select the most likely ball using multi-factor scoring.

        Scores each candidate on:
        - confidence (detection quality)
        - motion continuity (distance from previous position)
        - bounding box size sanity
        Rejects impossible jumps.
        """
        if not active_tracks:
            return {"frame": frame_idx, "ball": None}

        candidates = []
        for trk in active_tracks:
            bbox = trk["bbox"]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            area = w * h

            # Size sanity check
            if area < self.min_bbox_area or area > self.max_bbox_area:
                continue

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # Multi-factor score
            conf_score = trk["score"]
            continuity_score = 1.0

            if prev_ball is not None:
                dx = cx - prev_ball["x"]
                dy = cy - prev_ball["y"]
                displacement = np.sqrt(dx * dx + dy * dy)

                # Reject impossible jumps
                if displacement > self.max_displacement_px:
                    continue

                # Continuity: closer to previous position = higher score
                # Normalized: 0 displacement = 1.0, max_speed = 0.0
                continuity_score = max(0.0, 1.0 - displacement / self.max_speed_px)

            # Combined score: weighted sum
            combined = 0.6 * conf_score + 0.4 * continuity_score
            candidates.append((combined, cx, cy, bbox, trk))

        if not candidates:
            return {"frame": frame_idx, "ball": None}

        best_score, cx, cy, bbox, trk = max(candidates, key=lambda c: c[0])

        return {
            "frame": frame_idx,
            "ball": {
                "x": cx,
                "y": cy,
                "bbox": bbox,
                "confidence": trk["score"],
                "track_id": trk["track_id"],
            },
        }
