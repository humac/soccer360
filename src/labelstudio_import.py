"""Helpers for converting hard-frame manifests into Label Studio tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DETECTION_IMG_W = 1920
DETECTION_IMG_H = 960


def _task_prediction_from_bbox(bbox: list[float] | tuple[float, float, float, float]) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    return {
        "model_version": "ball_detector",
        "result": [{
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": round((x1 / DETECTION_IMG_W) * 100, 2),
                "y": round((y1 / DETECTION_IMG_H) * 100, 2),
                "width": round(((x2 - x1) / DETECTION_IMG_W) * 100, 2),
                "height": round(((y2 - y1) / DETECTION_IMG_H) * 100, 2),
                "rectanglelabels": ["ball"],
            },
        }],
    }


def build_tasks(match_name: str, frames_dir: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Build Label Studio tasks from exported hard frames.

    Supports both current and legacy manifest frame entries:
    - current active-learning: bbox/conf
    - legacy hard-frames: predicted_bbox/predicted_confidence

    When both bbox keys are present, predicted_bbox is preferred to preserve
    legacy pre-annotation intent. If no bbox is present, a task is still emitted
    without predictions.
    """
    frame_meta: dict[int, dict[str, Any]] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for frame in manifest.get("frames", []):
            frame_index = frame.get("frame_index")
            if frame_index is not None:
                frame_meta[int(frame_index)] = frame

    tasks: list[dict[str, Any]] = []
    for img in sorted(frames_dir.glob("frame_*.jpg")):
        frame_idx = int(img.stem.split("_")[1])
        task: dict[str, Any] = {
            "data": {
                "image": f"/data/local-files/?d=labeling/{match_name}/frames/{img.name}",
                "frame_index": frame_idx,
                "match_name": match_name,
            }
        }

        meta = frame_meta.get(frame_idx, {})
        bbox = meta.get("predicted_bbox") or meta.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            task["predictions"] = [_task_prediction_from_bbox(bbox)]

        tasks.append(task)

    return tasks


def write_tasks_json(match_name: str, frames_dir: Path, output_dir: Path, manifest_path: Path) -> Path:
    """Write Label Studio tasks.json and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "tasks.json"
    output_file.write_text(json.dumps(build_tasks(match_name, frames_dir, manifest_path), indent=2))
    return output_file
