"""V1 active learning frame export.

Exports challenging frames for human labeling based on detection
quality signals: low confidence, lost ball runs, and jump rejections.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .utils import (
    VideoMeta,
    extract_frame,
    load_detections_jsonl,
    load_json,
    write_json,
)

logger = logging.getLogger("soccer360.active_learning")


class ActiveLearningExporter:
    """V1 active learning frame export with three-trigger identification."""

    def __init__(self, config: dict):
        al_cfg = config.get("active_learning", {})

        self.enabled = al_cfg.get("enabled", True)
        self.export_dir = Path(al_cfg.get("export_dir", "/tank/labeling"))
        self.export_max_frames = al_cfg.get("export_max_frames", 600)
        self.export_every_n = al_cfg.get("export_every_n_frames", 2)
        self.low_conf_min = al_cfg.get("low_conf_min", 0.20)
        self.low_conf_max = al_cfg.get("low_conf_max", 0.50)
        self.lost_run_frames = al_cfg.get("lost_run_frames", 15)
        self.jump_trigger_px = al_cfg.get("jump_trigger_px", 200)

    @staticmethod
    def _is_low_conf_only(candidate: dict) -> bool:
        """True when candidate was triggered only by low confidence."""
        return set(candidate.get("triggers", [])) == {"low_conf"}

    @staticmethod
    def _is_rare(candidate: dict) -> bool:
        """Rare triggers should bypass modulo gating."""
        triggers = set(candidate.get("triggers", []))
        return "lost_run" in triggers or "jump_reject" in triggers

    @staticmethod
    def _evenly_sample(candidates: list[dict], k: int) -> list[dict]:
        """Deterministically sample k items spread across sorted candidates."""
        n = len(candidates)
        if k <= 0:
            return []
        if k >= n:
            return list(candidates)

        sampled: list[dict] = []
        for i in range(k):
            idx = ((2 * i + 1) * n) // (2 * k)
            sampled.append(candidates[idx])
        return sampled

    def run(
        self,
        video_path: str,
        meta: VideoMeta,
        detections_path: Path,
        tracks_path: Path,
        work_dir: Path,
        tracking_events: list[dict] | None = None,
        mode: str = "normal",
    ):
        """Identify and export hard frames for labeling.

        Args:
            video_path: Source video file.
            meta: Video metadata.
            detections_path: Path to detections.jsonl.
            tracks_path: Path to tracks.json.
            work_dir: Pipeline working directory (manifest copied here).
            tracking_events: Events from BallStabilizer for jump/speed triggers.
            mode: Pipeline operating mode ("normal" or "no_detect").
        """
        if not self.enabled:
            logger.info("Active learning export disabled")
            return

        if mode == "no_detect":
            logger.info("Skipping active learning in NO_DETECT mode")
            return

        detections = load_detections_jsonl(detections_path)
        tracks = load_json(tracks_path)
        match_name = Path(video_path).stem

        candidates = self._identify_candidates(
            detections, tracks, meta, tracking_events or []
        )

        if not candidates:
            logger.info("No hard frames to export for %s", match_name)
            return

        logger.info(
            "Identified %d hard frame candidates for %s", len(candidates), match_name
        )

        ordered = sorted(candidates, key=lambda c: c["frame_index"])
        rare = [c for c in ordered if self._is_rare(c)]
        dense = [c for c in ordered if self._is_low_conf_only(c)]

        # Apply modulo gating only to low_conf-only candidates.
        if self.export_every_n <= 1:
            dense_gated = dense
        else:
            dense_gated = [
                c for c in dense if c["frame_index"] % self.export_every_n == 0
            ]

        cap = max(int(self.export_max_frames), 0)
        selected: list[dict]
        if cap <= 0:
            selected = []
        elif len(rare) >= cap:
            selected = self._evenly_sample(rare, cap)
        else:
            remaining = cap - len(rare)
            selected = rare + self._evenly_sample(dense_gated, remaining)

        selected = sorted(selected, key=lambda c: c["frame_index"])
        if not selected:
            logger.info("No candidates passed gating for %s", match_name)
            return

        # Export frames
        output_dir = self.export_dir / match_name
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        exported = []
        for hf in selected:
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
            "source_video": str(video_path),
            "total_candidates": len(candidates),
            "exported_count": len(exported),
            "config": {
                "low_conf_range": [self.low_conf_min, self.low_conf_max],
                "lost_run_frames": self.lost_run_frames,
                "jump_trigger_px": self.jump_trigger_px,
                "export_every_n": self.export_every_n,
                "export_max": self.export_max_frames,
            },
            "frames": exported,
        }

        manifest_path = output_dir / "hard_frames.json"
        write_json(manifest, manifest_path)

        # Copy manifest to work_dir for artifact preservation
        write_json(manifest, work_dir / "hard_frames.json")

        logger.info(
            "Exported %d hard frames to %s (manifest: %s)",
            len(exported), frames_dir, manifest_path,
        )

    def _identify_candidates(
        self,
        detections: list[dict],
        tracks: list[dict],
        meta: VideoMeta,
        tracking_events: list[dict],
    ) -> list[dict]:
        """Identify candidate hard frames from three triggers.

        Triggers:
          1. low_conf: detection confidence in [low_conf_min, low_conf_max]
          2. lost_run: consecutive ball=None streak >= lost_run_frames
             (ONE representative frame per streak at threshold crossing)
          3. jump_reject: tracking event with distance_px >= jump_trigger_px
        """
        candidates: dict[int, dict] = {}

        # Trigger 1: Low confidence detections
        for det in detections:
            conf = det.get("conf", det.get("confidence", 0.0))
            if self.low_conf_min <= conf <= self.low_conf_max:
                idx = det.get("frame_index", det.get("frame", -1))
                entry = candidates.setdefault(idx, {
                    "frame_index": idx,
                    "time_sec": round(idx / meta.fps, 3) if meta.fps > 0 else 0.0,
                    "triggers": [],
                })
                if "low_conf" not in entry["triggers"]:
                    entry["triggers"].append("low_conf")
                    entry["conf"] = conf
                    entry["bbox"] = det.get("bbox_xyxy", det.get("bbox"))

        # Trigger 2: Lost ball runs (one representative frame per streak)
        streak = 0
        prev_frame: int | None = None
        for track in tracks:
            frame_idx = int(track.get("frame", -1))

            if prev_frame is not None and frame_idx != prev_frame + 1:
                streak = 0

            if track.get("ball") is None:
                streak += 1
                # Export exactly at threshold crossing
                if streak == self.lost_run_frames:
                    entry = candidates.setdefault(frame_idx, {
                        "frame_index": frame_idx,
                        "time_sec": round(
                            frame_idx / meta.fps, 3
                        ) if meta.fps > 0 else 0.0,
                        "triggers": [],
                    })
                    if "lost_run" not in entry["triggers"]:
                        entry["triggers"].append("lost_run")
                        entry["streak_length"] = streak
            else:
                streak = 0
            prev_frame = frame_idx

        # Trigger 3: Jump/speed rejection events (gated by jump_trigger_px)
        for event in tracking_events:
            trigger = event.get("trigger", "")
            distance = event.get("distance_px", 0.0)
            if trigger in ("jump_reject", "speed_reject"):
                if distance >= self.jump_trigger_px:
                    idx = event["frame"]
                    entry = candidates.setdefault(idx, {
                        "frame_index": idx,
                        "time_sec": round(
                            idx / meta.fps, 3
                        ) if meta.fps > 0 else 0.0,
                        "triggers": [],
                    })
                    if "jump_reject" not in entry["triggers"]:
                        entry["triggers"].append("jump_reject")
                        entry["distance_px"] = distance

        return sorted(candidates.values(), key=lambda c: c["frame_index"])
