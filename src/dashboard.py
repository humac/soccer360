"""FastAPI monitoring dashboard for Soccer360 pipeline.

Serves a single-page HTML dashboard and provides a REST API + SSE
event stream for real-time pipeline monitoring and training management.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import errno
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import tempfile
import zipfile

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:  # pragma: no cover - fallback for lightweight host test envs
    from starlette.responses import StreamingResponse

    def _format_sse_payload(event: dict) -> bytes:
        lines = []
        event_name = event.get("event")
        event_id = event.get("id")
        data = event.get("data", "")

        if event_name:
            lines.append(f"event: {event_name}")
        if event_id:
            lines.append(f"id: {event_id}")

        data_text = str(data)
        for line in data_text.splitlines() or [""]:
            lines.append(f"data: {line}")

        return ("\n".join(lines) + "\n\n").encode("utf-8")

    class EventSourceResponse(StreamingResponse):
        """Minimal SSE-compatible fallback used when sse-starlette is unavailable."""

        def __init__(self, content, *args, **kwargs):
            async def _stream():
                async for event in content:
                    yield _format_sse_payload(event)

            headers = dict(kwargs.pop("headers", {}) or {})
            headers.setdefault("Cache-Control", "no-cache")
            headers.setdefault("Connection", "keep-alive")
            super().__init__(
                _stream(),
                *args,
                headers=headers,
                media_type="text/event-stream",
                **kwargs,
            )

from .events import EventStore, create_event_store
from .metrics import cpu_ram_snapshot, gpu_utilization_snapshot
from .detector import (
    is_v1_model_selection_locked_by_config,
    load_v1_runtime_model_selection,
    resolve_v1_model_path_and_source,
    save_v1_runtime_model_selection,
)
from .watcher import ProcessedIngestStore

logger = logging.getLogger("soccer360.dashboard")

APP_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")
DETECTION_SETTINGS_SECTIONS = [
    {
        "key": "detection",
        "title": "Detection",
        "fields": [
            {"path": "detection.path", "label": "YOLO pipeline model path", "help": "Configured model path for the YOLO Detection Pipeline."},
            {"path": "detection.classes", "label": "Classes", "help": "YOLO class IDs used during the YOLO Detection Pipeline."},
            {"path": "detection.conf", "label": "Confidence threshold", "help": "Minimum confidence for accepted detections in the YOLO Detection Pipeline."},
            {"path": "detection.iou", "label": "NMS IoU threshold", "help": "Non-max suppression overlap threshold for the YOLO Detection Pipeline."},
            {"path": "detection.img_size", "label": "Image size", "help": "Inference image size for the YOLO Detection Pipeline."},
            {"path": "detection.max_det", "label": "Max detections", "help": "Maximum detections retained per frame."},
            {"path": "detection.half", "label": "FP16 enabled", "help": "Whether half precision is enabled for the YOLO Detection Pipeline."},
            {"path": "detection.device", "label": "Detection device", "help": "Torch device string used for the YOLO Detection Pipeline."},
            {"path": "detector.batch_size", "label": "Streaming batch size", "help": "Main detector batch size used during streaming inference."},
            {"path": "detector.batch_size_tensorrt", "label": "TensorRT batch size", "help": "Streaming batch size used for TensorRT inference."},
            {"path": "detector.resolution", "label": "Detector resolution", "help": "Frame resolution passed into the main detector."},
            {"path": "detector.confidence_threshold", "label": "Streaming confidence threshold", "help": "Confidence threshold for the main detector path."},
            {"path": "detector.nms_iou_threshold", "label": "Streaming NMS IoU threshold", "help": "IoU threshold for detector-side non-max suppression."},
            {"path": "detector.process_every_n_frames", "label": "Process every N frames", "help": "Frame-skip factor for detector inference."},
            {"path": "detector.tiling.enabled", "label": "Tiling enabled", "help": "Whether detector tiling is enabled."},
            {"path": "detector.tiling.tiles", "label": "Tiling tiles", "help": "Number of detector tiles when tiling is enabled."},
            {"path": "detector.tiling.overlap", "label": "Tiling overlap", "help": "Tile overlap ratio for detector tiling."},
        ],
    },
    {
        "key": "field_of_interest",
        "title": "Field of Interest",
        "fields": [
            {"path": "field_of_interest.enabled", "label": "Enabled", "help": "Whether the field-of-interest crop gate is active."},
            {"path": "field_of_interest.center_mode", "label": "Center mode", "help": "Fixed or auto center selection mode."},
            {"path": "field_of_interest.center_yaw_deg", "label": "Center yaw (deg)", "help": "Fixed center yaw angle for the playable field."},
            {"path": "field_of_interest.yaw_window_deg", "label": "Yaw window (deg)", "help": "Horizontal field-of-interest window width."},
            {"path": "field_of_interest.pitch_min_deg", "label": "Pitch min (deg)", "help": "Lower pitch bound for accepted detections."},
            {"path": "field_of_interest.pitch_max_deg", "label": "Pitch max (deg)", "help": "Upper pitch bound for accepted detections."},
            {"path": "field_of_interest.auto_sample_seconds", "label": "Auto sample seconds", "help": "Duration used to estimate auto-centered field yaw."},
            {"path": "field_of_interest.auto_min_conf", "label": "Auto-center min confidence", "help": "Minimum detection confidence for auto-centering samples."},
        ],
    },
    {
        "key": "filters",
        "title": "Ball Stabilization / Filters",
        "fields": [
            {"path": "filters.min_y_frac", "label": "Min Y fraction", "help": "Reject detections above this normalized vertical boundary."},
            {"path": "filters.max_y_frac", "label": "Max Y fraction", "help": "Reject detections below this normalized vertical boundary."},
            {"path": "filters.max_jump_px", "label": "Max jump pixels", "help": "Maximum allowed position jump before rejection."},
            {"path": "filters.max_speed_px_per_s", "label": "Max speed px/sec", "help": "Maximum allowed motion speed before rejection."},
            {"path": "filters.jump_max_gap_frames", "label": "Jump reset gap frames", "help": "Gap after which jump anchoring resets for reacquisition."},
            {"path": "tracking.ema_alpha", "label": "Tracking EMA alpha", "help": "Temporal smoothing factor for stabilized ball positions."},
            {"path": "tracking.require_persistence", "label": "Persistence requirement", "help": "Number of detections required before accepting a track."},
            {"path": "tracking.window", "label": "Persistence window", "help": "Rolling window used for persistence gating."},
        ],
    },
    {
        "key": "player_cluster",
        "title": "Player Detection & Clustering",
        "fields": [
            {"path": "center_of_play.enabled", "label": "Enabled", "help": "Whether the player cluster is used to guide framing."},
            {"path": "center_of_play.player_class", "label": "Player class", "help": "Detector class ID used for player detections."},
            {"path": "center_of_play.min_player_conf", "label": "Min player confidence", "help": "Confidence threshold for player detections in clustering."},
            {"path": "center_of_play.trim_fraction", "label": "Trim fraction", "help": "Outlier trim ratio before computing player centroid."},
            {"path": "center_of_play.min_players", "label": "Min players", "help": "Minimum player detections needed for a valid cluster."},
            {"path": "center_of_play.ball_blend_weight", "label": "Ball blend weight", "help": "How much the player cluster influences camera aim when ball exists."},
            {"path": "center_of_play.ema_alpha", "label": "Cluster EMA alpha", "help": "Temporal smoothing factor for the cluster centroid."},
            {"path": "center_of_play.fov_from_spread", "label": "Adaptive FOV", "help": "Whether FOV widens based on player spread."},
            {"path": "center_of_play.spread_max_fov", "label": "Spread max FOV", "help": "Maximum FOV when players are widely spread."},
            {"path": "center_of_play.spread_min_deg", "label": "Spread min degrees", "help": "Spread below this uses the minimum framing FOV."},
            {"path": "center_of_play.spread_max_deg", "label": "Spread max degrees", "help": "Spread above this uses the maximum adaptive FOV."},
        ],
    },
    {
        "key": "camera",
        "title": "Camera / Auto-Follow",
        "fields": [
            {"path": "camera.max_pan_speed_deg_per_sec", "label": "Max pan speed", "help": "Normal maximum camera pan speed."},
            {"path": "camera.max_fast_pan_speed_deg_per_sec", "label": "Fast pan speed", "help": "Faster pan speed used for rapid action shifts."},
            {"path": "camera.ema_alpha", "label": "Camera EMA alpha", "help": "Smoothing factor for camera path updates."},
            {"path": "camera.default_fov", "label": "Default FOV", "help": "Default follow-camera field of view."},
            {"path": "camera.min_fov", "label": "Min FOV", "help": "Tightest follow-camera field of view."},
            {"path": "camera.max_fov", "label": "Max FOV", "help": "Widest follow-camera field of view."},
            {"path": "camera.lost_coast_frames", "label": "Lost coast frames", "help": "Frames to hold motion when the ball is temporarily lost."},
            {"path": "camera.lost_drift_frames", "label": "Lost drift frames", "help": "Frames to drift back toward center after longer ball loss."},
            {"path": "camera.field_center_yaw_deg", "label": "Field center yaw", "help": "Nominal field-center yaw angle."},
            {"path": "camera.field_center_pitch_deg", "label": "Field center pitch", "help": "Nominal field-center pitch angle."},
            {"path": "camera.deadband_deg", "label": "Deadband", "help": "Ignore tiny camera movements to avoid jitter."},
            {"path": "camera.velocity_threshold_deg_per_sec", "label": "Velocity threshold", "help": "Minimum camera movement speed before panning starts."},
            {"path": "camera.lost_fov_widen", "label": "Lost FOV widen", "help": "Whether to widen the FOV immediately when the ball is lost."},
            {"path": "camera.fov_ema_alpha", "label": "FOV EMA alpha", "help": "Smoothing factor for zoom transitions."},
            {"path": "camera.kalman.process_noise", "label": "Kalman process noise", "help": "Process noise for camera path Kalman filtering."},
            {"path": "camera.kalman.measurement_noise", "label": "Kalman measurement noise", "help": "Measurement noise for camera path Kalman filtering."},
        ],
    },
    {
        "key": "reframer",
        "title": "Reframer / Output",
        "fields": [
            {"path": "reframer.output_resolution", "label": "Output resolution", "help": "Final broadcast output resolution."},
            {"path": "reframer.source_downscale", "label": "Source downscale", "help": "Optional pre-reframe source downscale."},
            {"path": "reframer.num_workers", "label": "Render workers", "help": "Parallel reframer worker count."},
            {"path": "reframer.interpolation", "label": "Interpolation", "help": "Interpolation method used during equirectangular reprojection."},
            {"path": "reframer.overlap_sec", "label": "Segment overlap (sec)", "help": "Overlap between render segments for clean joins."},
            {"path": "reframer.tactical_fov", "label": "Tactical FOV", "help": "Field of view used for the tactical wide output."},
            {"path": "reframer.tactical_yaw", "label": "Tactical yaw", "help": "Yaw used for the tactical wide output."},
            {"path": "reframer.tactical_pitch", "label": "Tactical pitch", "help": "Pitch used for the tactical wide output."},
        ],
    },
    {
        "key": "highlights",
        "title": "Highlights",
        "fields": [
            {"path": "highlights.speed_percentile", "label": "Speed percentile", "help": "Ball-speed percentile used as a highlight trigger."},
            {"path": "highlights.direction_change_deg", "label": "Direction change (deg)", "help": "Direction-change threshold used by highlight heuristics."},
            {"path": "highlights.goal_box_regions", "label": "Goal box regions", "help": "Normalized goal-area regions used for highlight scoring."},
            {"path": "highlights.pre_margin_sec", "label": "Pre-roll margin", "help": "Seconds added before a highlight event."},
            {"path": "highlights.post_margin_sec", "label": "Post-roll margin", "help": "Seconds added after a highlight event."},
            {"path": "highlights.min_clip_gap_sec", "label": "Min clip gap", "help": "Minimum spacing before clips are merged or separated."},
            {"path": "highlights.min_clip_duration_sec", "label": "Min clip duration", "help": "Minimum exported highlight duration."},
            {"path": "highlights.cluster_convergence_window", "label": "Cluster convergence window", "help": "Frame window for cluster convergence detection."},
            {"path": "highlights.cluster_convergence_deg", "label": "Cluster convergence degrees", "help": "Angular threshold for convergence-based highlight detection."},
            {"path": "highlights.cluster_velocity_window", "label": "Cluster velocity window", "help": "Frame window for cluster-velocity scoring."},
            {"path": "highlights.cluster_velocity_deg_per_sec", "label": "Cluster velocity deg/sec", "help": "Velocity threshold for cluster-based highlights."},
            {"path": "highlights.cluster_goal_zone_regions", "label": "Cluster goal-zone regions", "help": "Optional cluster-specific goal-zone regions."},
            {"path": "highlights.cluster_density_percentile", "label": "Cluster density percentile", "help": "Density percentile threshold for cluster scoring."},
            {"path": "highlights.score_weights.speed", "label": "Score weight: speed", "help": "Relative score weight for ball speed events."},
            {"path": "highlights.score_weights.goal_box", "label": "Score weight: goal box", "help": "Relative score weight for goal-box proximity."},
            {"path": "highlights.score_weights.direction_change", "label": "Score weight: direction change", "help": "Relative score weight for abrupt ball direction changes."},
            {"path": "highlights.score_weights.cluster_convergence", "label": "Score weight: cluster convergence", "help": "Relative score weight for player convergence."},
            {"path": "highlights.score_weights.cluster_velocity", "label": "Score weight: cluster velocity", "help": "Relative score weight for player-cluster speed."},
            {"path": "highlights.score_weights.cluster_goal_zone", "label": "Score weight: cluster goal zone", "help": "Relative score weight for cluster goal-zone pressure."},
            {"path": "highlights.score_weights.cluster_density", "label": "Score weight: cluster density", "help": "Relative score weight for dense player-cluster moments."},
            {"path": "highlights.combined_signal_bonus", "label": "Combined signal bonus", "help": "Bonus score when multiple signals fire together."},
            {"path": "highlights.min_clip_score", "label": "Min clip score", "help": "Minimum total score required to export a highlight."},
            {"path": "highlights.max_clips", "label": "Max clips", "help": "Maximum number of highlight clips to export."},
        ],
    },
    {
        "key": "active_learning",
        "title": "Active Learning",
        "fields": [
            {"path": "active_learning.enabled", "label": "Enabled", "help": "Whether hard-frame export is enabled during processing."},
            {"path": "active_learning.export_max_frames", "label": "Max exported frames", "help": "Maximum hard frames exported per match."},
            {"path": "active_learning.export_every_n_frames", "label": "Export every N frames", "help": "Only export every Nth candidate hard frame."},
            {"path": "active_learning.low_conf_min", "label": "Low-confidence minimum", "help": "Lower bound of the low-confidence export band."},
            {"path": "active_learning.low_conf_max", "label": "Low-confidence maximum", "help": "Upper bound of the low-confidence export band."},
            {"path": "active_learning.lost_run_frames", "label": "Lost-run frames", "help": "Consecutive lost-ball frames before exporting a hard frame."},
            {"path": "active_learning.jump_trigger_px", "label": "Jump trigger pixels", "help": "Jump-distance threshold for exporting a hard frame."},
        ],
    },
]


def _scan_labeling_status(labeling_dir: Path) -> dict:
    """Scan labeling directory for matches with exported frames and labels."""
    matches = []
    total_frames = 0
    total_labeled = 0

    if not labeling_dir.is_dir():
        return {"matches": [], "total_frames": 0, "total_labeled": 0, "dataset_ready": False}

    for match_dir in sorted(labeling_dir.iterdir()):
        if not match_dir.is_dir() or match_dir.name in ("dataset",):
            continue

        frames_dir = match_dir / "frames"
        labels_dir = match_dir / "labels"

        frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
        label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

        if frame_count > 0:
            tasks_json = match_dir / "labelstudio" / "tasks.json"
            tasks_count = 0
            if tasks_json.exists():
                try:
                    tasks_count = len(json.loads(tasks_json.read_text()))
                except Exception:
                    pass
            matches.append({
                "name": match_dir.name,
                "frames": frame_count,
                "labeled": label_count,
                "pct_labeled": round(label_count / frame_count * 100, 1) if frame_count > 0 else 0,
                "tasks_imported": tasks_json.exists(),
                "tasks_count": tasks_count,
            })
            total_frames += frame_count
            total_labeled += label_count

    dataset_yaml = labeling_dir / "dataset" / "dataset.yaml"

    return {
        "matches": matches,
        "total_frames": total_frames,
        "total_labeled": total_labeled,
        "dataset_ready": dataset_yaml.exists(),
        "dataset_yaml": str(dataset_yaml) if dataset_yaml.exists() else None,
    }


def _get_nested_config_value(config: dict, path: str):
    """Safely fetch a dotted config path."""
    current = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _build_detection_settings_payload(config: dict, models_dir: Path) -> dict:
    """Return dashboard payload describing effective video-processing settings."""
    inference = _inference_model_status(config, models_dir)
    groups = [
        {
            "key": "ingest_model",
            "title": "Ingest Model",
            "fields": [
                {
                    "label": "Resolved model path",
                    "config_path": "runtime.resolved_path",
                    "help": "The effective model path future ingest jobs will resolve today.",
                    "value": inference.get("resolved_path"),
                },
                {
                    "label": "Resolved source",
                    "config_path": "runtime.resolved_source",
                    "help": "Which resolver source selected the effective ingest model.",
                    "value": inference.get("resolved_source"),
                },
                {
                    "label": "Selection mode",
                    "config_path": "runtime.selection_mode",
                    "help": "Dashboard-managed ingest model selection mode. Change this in Staging.",
                    "value": inference.get("selection_mode"),
                },
                {
                    "label": "Pinned model path",
                    "config_path": "runtime.selected_path",
                    "help": "Pinned path when runtime ingest selection is set to pinned mode.",
                    "value": inference.get("selected_path"),
                },
                {
                    "label": "Selection file",
                    "config_path": "detector.runtime_override_path",
                    "help": "Path where dashboard-managed ingest model selection is stored.",
                    "value": inference.get("selection_path"),
                },
                {
                    "label": "Config lock active",
                    "config_path": "detector.model_path",
                    "help": "True when detector.model_path explicitly locks runtime selection changes.",
                    "value": inference.get("config_locked"),
                },
            ],
        }
    ]

    for section in DETECTION_SETTINGS_SECTIONS:
        groups.append({
            "key": section["key"],
            "title": section["title"],
            "fields": [
                {
                    "label": field["label"],
                    "config_path": field["path"],
                    "help": field["help"],
                    "value": _get_nested_config_value(config, field["path"]),
                }
                for field in section["fields"]
            ],
        })

    return {
        "readonly": True,
        "scope": "future_ingest_jobs",
        "title": "Detection Settings",
        "description": "Readonly view of the effective configuration used for future video-processing jobs.",
        "note": "Ingest model selection is managed in the Staging section of the main dashboard.",
        "groups": groups,
    }


def _build_dataset_from_labels(
    labeling_dir: Path,
    val_ratio: float = 0.2,
    output_dir: Path | None = None,
) -> dict:
    """Build a YOLO dataset from Label Studio frame/label exports."""
    if not labeling_dir.is_dir():
        raise ValueError(f"Labeling directory not found: {labeling_dir}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"Validation ratio must be between 0 and 1, got {val_ratio}")

    output_dir = output_dir or (labeling_dir / "dataset")

    pairs: list[tuple[Path, Path, str]] = []
    match_counts: dict[str, int] = {}
    for match_dir in sorted(labeling_dir.iterdir()):
        if not match_dir.is_dir() or match_dir.name == "dataset":
            continue

        labels_dir = match_dir / "labels"
        frames_dir = match_dir / "frames"
        if not labels_dir.exists() or not frames_dir.exists():
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            if label_file.name.lower() == "classes.txt":
                continue
            image_file = _resolve_frame_for_label(frames_dir, label_file.name)
            if image_file.exists():
                pairs.append((image_file, label_file, match_dir.name))
                match_counts[match_dir.name] = match_counts.get(match_dir.name, 0) + 1

    if not pairs:
        diagnostics = _labeling_pair_diagnostics(labeling_dir)
        raise ValueError(
            "No image/label pairs found. Expected "
            "<labeling>/<match>/frames/frame_XXXXXX.jpg and "
            "<labeling>/<match>/labels/frame_XXXXXX.txt. "
            f"Found {diagnostics['frame_count']} frame(s), "
            f"{diagnostics['label_count']} label file(s), "
            f"{diagnostics['matched_count']} matched pair(s). "
            f"Sample unmatched labels: {diagnostics['sample_unmatched_labels'] or 'none'}."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)

    random.Random(42).shuffle(pairs)
    split_idx = max(1, int(len(pairs) * (1 - val_ratio)))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    for split_name, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        img_dir = output_dir / split_name / "images"
        lbl_dir = output_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path, match_name in split_pairs:
            dest_name = f"{match_name}_{img_path.name}"
            shutil.copy2(img_path, img_dir / dest_name)
            shutil.copy2(lbl_path, lbl_dir / dest_name.replace(".jpg", ".txt"))

    yaml_content = f"""# Soccer360 Ball Detection Dataset
# Auto-generated by dashboard dataset builder

path: {output_dir}
train: train/images
val: val/images

nc: 1
names:
  0: ball
"""
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(yaml_content)

    return {
        "dataset_yaml": dataset_yaml,
        "output_dir": output_dir,
        "match_counts": match_counts,
        "train_count": len(train_pairs),
        "val_count": len(val_pairs),
        "total_count": len(pairs),
        "val_ratio": val_ratio,
    }


def _load_json_file(path: Path) -> dict:
    """Load a JSON object, returning {} for missing/invalid files."""
    if not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def _validate_training_base_model_choice(base_model: str | None, models_dir: Path) -> str | None:
    """Validate a requested training base model path against local allowed roots."""
    if not base_model:
        return None

    candidate = Path(str(base_model))
    if not candidate.is_absolute():
        candidate = models_dir / candidate

    try:
        resolved = candidate.resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base model path: {exc}")

    if not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"Base model not found: {resolved}")

    allowed_roots = []
    for root in (models_dir, Path("/app/models"), Path("/app")):
        try:
            allowed_roots.append(root.resolve())
        except Exception:
            continue

    try:
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=400, detail="Base model path is outside allowed model roots")
    except AttributeError:
        resolved_str = str(resolved)
        allowed = False
        for root in allowed_roots:
            root_str = str(root)
            if resolved_str == root_str or resolved_str.startswith(root_str + os.sep):
                allowed = True
                break
        if not allowed:
            raise HTTPException(status_code=400, detail="Base model path is outside allowed model roots")

    return str(resolved)


def _path_identity(path: Path | str | None) -> str:
    """Best-effort stable identity for comparing paths across mount aliases."""
    if not path:
        return ""
    candidate = Path(str(path))
    try:
        return str(candidate.resolve())
    except Exception:
        return str(candidate)


def _validate_deletable_model_path(
    model_path: str | None,
    models_dir: Path,
    *,
    protected_paths: set[str],
) -> Path:
    """Validate a requested model deletion target and enforce safety rails."""
    if not model_path:
        raise HTTPException(status_code=400, detail="Model path is required")

    candidate = Path(str(model_path))
    try:
        resolved = candidate.resolve()
        models_root = models_dir.resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid model path: {exc}")

    try:
        if not resolved.is_relative_to(models_root):
            raise HTTPException(status_code=400, detail="Only local models under the models directory can be deleted")
    except AttributeError:
        resolved_str = str(resolved)
        root_str = str(models_root)
        if not (resolved_str == root_str or resolved_str.startswith(root_str + os.sep)):
            raise HTTPException(status_code=400, detail="Only local models under the models directory can be deleted")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Model file not found")
    if resolved.name == "ball_best.pt":
        raise HTTPException(status_code=409, detail="Cannot delete the active inference model ball_best.pt")
    if _path_identity(resolved) in protected_paths:
        raise HTTPException(status_code=409, detail="Cannot delete a configured model that is protected by the dashboard")

    return resolved


def _candidate_frame_names_for_label(label_name: str) -> list[str]:
    """Return plausible frame filenames for a label export name."""
    stem = Path(label_name).stem
    candidates: list[str] = []
    seen: set[str] = set()

    def _add_stem(candidate_stem: str):
        if not candidate_stem:
            return
        for ext in FRAME_EXTENSIONS:
            filename = f"{candidate_stem}{ext}"
            key = filename.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(filename)

    _add_stem(stem)

    lower_stem = stem.lower()
    for suffix in ("_jpg", "_jpeg", "_png"):
        if lower_stem.endswith(suffix):
            _add_stem(stem[: -len(suffix)])

    frame_match = re.search(r"(frame_\d+)", stem, re.IGNORECASE)
    if frame_match:
        _add_stem(frame_match.group(1))

    return candidates


def _resolve_frame_for_label(frames_dir: Path, label_name: str) -> Path:
    """Resolve a label filename to a frame path, tolerating common export naming variants."""
    for candidate in _candidate_frame_names_for_label(label_name):
        frame_path = frames_dir / candidate
        if frame_path.is_file():
            return frame_path
    return frames_dir / Path(label_name).with_suffix(".jpg").name


def _labeling_pair_diagnostics(labeling_dir: Path) -> dict:
    """Summarize frame/label pairing status for build-dataset errors."""
    frame_count = 0
    label_count = 0
    matched_count = 0
    unmatched_labels: list[str] = []

    if not labeling_dir.is_dir():
        return {
            "frame_count": 0,
            "label_count": 0,
            "matched_count": 0,
            "sample_unmatched_labels": [],
        }

    for match_dir in sorted(labeling_dir.iterdir()):
        if not match_dir.is_dir() or match_dir.name == "dataset":
            continue

        frames_dir = match_dir / "frames"
        labels_dir = match_dir / "labels"
        if frames_dir.is_dir():
            frame_count += sum(1 for path in frames_dir.iterdir() if path.is_file())
        if not labels_dir.is_dir():
            continue

        for label_file in sorted(labels_dir.glob("*.txt")):
            if label_file.name.lower() == "classes.txt":
                continue
            label_count += 1
            frame_path = _resolve_frame_for_label(frames_dir, label_file.name)
            if frame_path.is_file():
                matched_count += 1
            elif len(unmatched_labels) < 5:
                unmatched_labels.append(f"{match_dir.name}/{label_file.name}")

    return {
        "frame_count": frame_count,
        "label_count": label_count,
        "matched_count": matched_count,
        "sample_unmatched_labels": unmatched_labels,
    }


def _validate_flat_name(value: str, field_name: str):
    """Reject path-like or hidden names for route/body parameters."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or value.startswith(".")
        or "/" in value
        or "\\" in value
        or ".." in value
        or Path(value).name != value
    ):
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}")


def _unique_file_path(parent: Path, filename: str) -> Path:
    """Return a collision-safe destination inside parent."""
    candidate = parent / filename
    if not candidate.exists():
        return candidate

    path = Path(filename)
    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter:02d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _safe_copy(src: Path, dst: Path, *, overwrite: bool = False):
    """Copy a file with atomic publish semantics."""
    src_size = src.stat().st_size
    tmp = dst.parent / f".{dst.name}.tmp-{os.getpid()}"

    try:
        shutil.copy2(str(src), str(tmp))
        if overwrite:
            os.replace(str(tmp), str(dst))
        else:
            try:
                os.link(str(tmp), str(dst))
                tmp.unlink()
            except FileExistsError:
                raise
            except OSError as exc:
                if exc.errno not in {errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
                    raise
                with tmp.open("rb") as src_f:
                    with dst.open("xb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)

        if dst.stat().st_size != src_size:
            raise RuntimeError(f"Size mismatch after copy: {src} -> {dst}")
    finally:
        if tmp.exists():
            tmp.unlink()


def _safe_move(src: Path, dst: Path, *, overwrite: bool = False):
    """Move a file with cross-filesystem fallback."""
    try:
        if overwrite:
            os.replace(str(src), str(dst))
        else:
            os.link(str(src), str(dst))
            src.unlink()
        return
    except FileExistsError:
        raise
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
            raise

    _safe_copy(src, dst, overwrite=overwrite)
    src.unlink()


def _resolve_processed_state_path(config: dict, processed_dir: Path) -> Path:
    """Resolve the watcher processed-state path using watcher semantics."""
    watcher_cfg = config.get("watcher", {})
    state_cfg = watcher_cfg.get("processed_state_file")
    state_base = processed_dir / ".state"

    if state_cfg:
        state_path = Path(state_cfg)
        if not state_path.is_absolute():
            state_path = state_base / state_path
        return state_path

    return state_base / "watcher_processed_ingest.json"


def _list_staging_files(
    staging_dir: Path,
    *,
    video_extensions: set[str],
    ignore_suffixes: set[str],
) -> list[dict]:
    """List eligible top-level staged video files."""
    if not staging_dir.is_dir():
        return []

    files = []
    for path in sorted(
        staging_dir.iterdir(),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    ):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in ignore_suffixes:
            continue
        if path.suffix.lower() not in video_extensions:
            continue

        stat = path.stat()
        files.append({
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_mb": round(stat.st_size / 1e6, 1),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })

    return files


def _resolve_dashboard_runtime_path(path_value: str, models_dir: Path) -> Path:
    """Resolve dashboard-supplied model paths consistently with runtime resolution."""
    candidate = Path(str(path_value))
    if candidate.is_absolute():
        return candidate
    return models_dir / candidate


def _collect_available_models(config: dict, models_dir: Path) -> list[dict]:
    """Collect unique available .pt models from local and configured locations."""
    configured_base = _path_identity(config.get("model", {}).get("base_model"))
    configured_detection = _path_identity(config.get("detection", {}).get("path"))
    configured_detector = _path_identity(config.get("detector", {}).get("model_path"))
    active_inference_path = _path_identity(
        _inference_model_status(config, models_dir).get("resolved_path")
    )
    protected_paths = {
        p for p in (
            configured_base,
            configured_detection,
            configured_detector,
            active_inference_path,
        ) if p
    }

    candidates: list[Path] = []
    if models_dir.is_dir():
        candidates.extend(sorted(models_dir.glob("**/*.pt")))

    for configured in (
        config.get("model", {}).get("base_model"),
        config.get("detection", {}).get("path"),
        config.get("detector", {}).get("model_path"),
        "/app/models/yolo26l.pt",
    ):
        if not configured:
            continue
        candidate = _resolve_dashboard_runtime_path(str(configured), models_dir)
        if candidate.is_file():
            candidates.append(candidate)

    seen: set[str] = set()
    entries = []
    models_root = _path_identity(models_dir)
    for path in candidates:
        identity = _path_identity(path)
        if not identity or identity in seen:
            continue
        seen.add(identity)

        is_local = False
        if models_root:
            try:
                is_local = Path(identity).is_relative_to(Path(models_root))
            except AttributeError:
                is_local = identity == models_root or identity.startswith(models_root + os.sep)

        entries.append({
            "path": identity,
            "name": Path(identity).name,
            "size_mb": round(Path(identity).stat().st_size / 1e6, 1),
            "is_active": Path(identity).name == "ball_best.pt",
            "is_inference_active": identity == active_inference_path,
            "is_configured_base": identity == configured_base,
            "is_configured_inference": identity in {configured_detection, configured_detector},
            "can_delete": (
                is_local
                and Path(identity).name != "ball_best.pt"
                and identity not in protected_paths
            ),
        })

    return sorted(
        entries,
        key=lambda item: (
            0 if item["is_active"] else 1,
            0 if item["is_configured_inference"] else 1,
            0 if item["is_configured_base"] else 1,
            item["name"].lower(),
        ),
    )


def _inference_model_status(config: dict, models_dir: Path) -> dict:
    """Return dashboard-friendly ingest model selection state."""
    selection = load_v1_runtime_model_selection(config)
    resolution_error = None
    try:
        resolved_path, resolved_source = resolve_v1_model_path_and_source(
            config,
            models_dir=str(models_dir),
        )
    except RuntimeError as exc:
        resolved_path, resolved_source = None, "unresolved"
        resolution_error = str(exc)
    return {
        "selection_mode": selection["mode"],
        "selected_path": selection.get("path"),
        "selection_path": selection["selection_path"],
        "resolved_path": resolved_path,
        "resolved_source": resolved_source,
        "resolution_error": resolution_error,
        "config_locked": is_v1_model_selection_locked_by_config(config),
    }


def create_app(config: dict | None = None) -> FastAPI:
    """Create and configure the FastAPI dashboard application."""
    config = config or {}
    store = create_event_store(config)

    app = FastAPI(title="Soccer360 Dashboard", version="0.1.0")

    # CORS: allow Label Studio (port 8080) to fetch images from dashboard (port 8088)
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Serve static files if directory exists
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # HTML entry point
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return HTMLResponse("<h1>Soccer360 Dashboard</h1><p>static/index.html not found</p>")
        return HTMLResponse(index_file.read_text())

    @app.get("/settings/detection", response_class=HTMLResponse)
    async def detection_settings_page():
        settings_file = STATIC_DIR / "detection_settings.html"
        if not settings_file.exists():
            return HTMLResponse("<h1>Soccer360 Dashboard</h1><p>static/detection_settings.html not found</p>")
        return HTMLResponse(settings_file.read_text())

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------

    @app.get("/api/status")
    async def status():
        return store.get_current_status()

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 50):
        return store.get_jobs(limit=limit)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        phases = store.get_phases(job_id)
        return {"job": job, "phases": phases}

    @app.get("/api/gpu")
    async def gpu_status():
        snap = gpu_utilization_snapshot()
        return snap or {"available": False}

    @app.get("/api/system")
    async def system_status():
        snap = cpu_ram_snapshot()
        return snap or {"available": False}

    @app.get("/api/decisions/pending")
    async def pending_decisions():
        return store.get_pending_decisions()

    @app.post("/api/decisions/{decision_id}/resolve")
    async def resolve_decision(decision_id: int, request: Request):
        body = await request.json()
        response = body.get("response", "")
        decision_status = body.get("status", "approved")
        if decision_status not in ("approved", "rejected"):
            raise HTTPException(status_code=400, detail="status must be 'approved' or 'rejected'")

        decision = store.get_decision(decision_id)
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")
        if decision["status"] != "pending":
            raise HTTPException(status_code=409, detail=f"Decision already resolved: {decision['status']}")

        store.resolve_decision(decision_id, response, status=decision_status)
        return {"ok": True, "decision_id": decision_id, "status": decision_status}

    # ------------------------------------------------------------------
    # Training / Active Learning
    # ------------------------------------------------------------------

    # In-memory training state (single-server, no persistence needed)
    _training_state = {"status": "idle", "log": [], "error": None}
    _training_lock = threading.Lock()

    paths_cfg = config.get("paths", {})
    ingest_dir = Path(paths_cfg.get("ingest", "/tank/ingest"))
    labeling_dir = Path(config.get("paths", {}).get("labeling", "/tank/labeling"))
    models_dir = Path(config.get("paths", {}).get("models", "/tank/models"))
    staging_dir = Path(paths_cfg.get("stagging", "/tank/stagging"))
    processed_dir = Path(paths_cfg.get("processed", "/tank/processed"))
    highlights_dir = Path(paths_cfg.get("highlights", "/tank/highlights"))
    processed_state_path = _resolve_processed_state_path(config, processed_dir)
    watcher_cfg = config.get("watcher", {})
    video_extensions = {
        str(ext).lower()
        for ext in watcher_cfg.get("extensions", [".mp4", ".insv", ".mov"])
    }
    ignore_suffixes = {
        str(ext).lower()
        for ext in watcher_cfg.get("ignore_suffixes", [".uploading", ".tmp", ".part"])
    }

    @app.get("/api/training/labeling-status")
    async def labeling_status():
        return _scan_labeling_status(labeling_dir)

    @app.post("/api/training/import-tasks/{match_name}")
    async def import_tasks(match_name: str, request: Request):
        """Generate Label Studio import tasks for a match."""
        if ".." in match_name or "/" in match_name:
            raise HTTPException(status_code=400, detail="Invalid match name")
        match_dir = labeling_dir / match_name
        frames_dir = match_dir / "frames"
        if not frames_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"No frames found for '{match_name}'. Process a video first.",
            )

        # Load manifest for predicted bboxes
        manifest_path = match_dir / "hard_frames.json"
        frame_meta = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                for f in manifest.get("frames", []):
                    frame_meta[f["frame_index"]] = f
            except Exception:
                pass

        # Build Label Studio task JSON
        # Use the Host header so URLs work from the browser's perspective
        host = request.headers.get("host", "localhost:8088")
        scheme = request.headers.get("x-forwarded-proto", "http")
        base_url = f"{scheme}://{host}"
        tasks = []
        img_w, img_h = 1920, 960  # detection resolution
        for img in sorted(frames_dir.glob("frame_*.jpg")):
            frame_idx = int(img.stem.split("_")[1])
            ls_path = f"{base_url}/api/labeling/frames/{match_name}/{img.name}"

            task: dict = {
                "data": {
                    "image": ls_path,
                    "frame_index": frame_idx,
                    "match_name": match_name,
                },
            }

            meta = frame_meta.get(frame_idx, {})
            bbox = meta.get("predicted_bbox") or meta.get("bbox")
            if bbox and len(bbox) == 4:
                x_pct = (bbox[0] / img_w) * 100
                y_pct = (bbox[1] / img_h) * 100
                w_pct = ((bbox[2] - bbox[0]) / img_w) * 100
                h_pct = ((bbox[3] - bbox[1]) / img_h) * 100
                task["predictions"] = [{
                    "model_version": "ball_detector",
                    "result": [{
                        "from_name": "label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "x": round(x_pct, 2),
                            "y": round(y_pct, 2),
                            "width": round(w_pct, 2),
                            "height": round(h_pct, 2),
                            "rectanglelabels": ["ball"],
                        },
                    }],
                }]
            tasks.append(task)

        # Write tasks.json
        ls_output_dir = match_dir / "labelstudio"
        ls_output_dir.mkdir(parents=True, exist_ok=True)
        tasks_file = ls_output_dir / "tasks.json"
        tasks_file.write_text(json.dumps(tasks, indent=2))

        return {"ok": True, "match": match_name, "tasks": len(tasks)}

    @app.post("/api/training/upload-labels/{match_name}")
    async def upload_labels(match_name: str, file: UploadFile):
        """Upload a ZIP of YOLO-format label .txt files for a match."""
        if ".." in match_name or "/" in match_name:
            raise HTTPException(status_code=400, detail="Invalid match name")
        match_dir = labeling_dir / match_name
        frames_dir = match_dir / "frames"
        if not frames_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"No frames found for '{match_name}'. Process a video first.",
            )

        # Save uploaded file to a temp location and extract
        labels_dir = match_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        extracted = 0

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=True) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()

            try:
                with zipfile.ZipFile(tmp.name, "r") as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        # Only extract .txt files (YOLO labels)
                        name = Path(info.filename).name
                        if not name.endswith(".txt") or name.startswith("."):
                            continue
                        if name.lower() == "classes.txt":
                            continue
                        # Path traversal safety: use only the basename
                        if ".." in info.filename:
                            continue
                        frame_path = _resolve_frame_for_label(frames_dir, name)
                        dest_name = (
                            frame_path.with_suffix(".txt").name
                            if frame_path.is_file()
                            else name
                        )
                        dest = labels_dir / dest_name
                        dest.write_bytes(zf.read(info.filename))
                        extracted += 1
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid ZIP file")

        if extracted == 0:
            raise HTTPException(
                status_code=400,
                detail="No .txt label files found in ZIP. Export from Label Studio in YOLO format.",
            )

        logger.info("Uploaded %d label files for match '%s'", extracted, match_name)
        return {"ok": True, "match": match_name, "labels_extracted": extracted}

    @app.get("/api/training/status")
    async def training_status():
        with _training_lock:
            return dict(_training_state)

    @app.get("/api/training/models")
    async def list_models():
        """List available models in /tank/models."""
        return _collect_available_models(config, models_dir)

    @app.post("/api/training/models/delete")
    async def delete_model(request: Request):
        """Delete a local old model file from the models directory."""
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        model_path = body.get("path")
        protected_paths = {
            p for p in (
                _path_identity(config.get("model", {}).get("base_model")),
                _path_identity(config.get("detection", {}).get("path")),
                _path_identity(config.get("detector", {}).get("model_path")),
                _path_identity(_inference_model_status(config, models_dir).get("resolved_path")),
            ) if p
        }

        with _training_lock:
            if _training_state["status"] in ("running", "building"):
                raise HTTPException(status_code=409, detail="Cannot delete models while training or dataset build is in progress")

        target = _validate_deletable_model_path(model_path, models_dir, protected_paths=protected_paths)
        target.unlink()

        current = target.parent
        try:
            models_root = models_dir.resolve()
        except Exception:
            models_root = models_dir

        while current != models_root:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

        logger.info("Deleted model file: %s", target)
        return {"ok": True, "deleted_path": str(target)}

    @app.get("/api/inference/model")
    async def get_inference_model():
        """Report the current ingest detection model selection and resolved path."""
        return _inference_model_status(config, models_dir)

    @app.get("/api/settings/detection")
    async def get_detection_settings():
        """Return readonly effective video-processing settings for future ingest jobs."""
        return _build_detection_settings_payload(config, models_dir)

    @app.post("/api/inference/model")
    async def set_inference_model(request: Request):
        """Set dashboard-managed ingest detection model selection."""
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        mode = str(body.get("mode", "config") or "config").strip().lower()
        model_path = body.get("path")

        if mode not in {"config", "auto", "pinned"}:
            raise HTTPException(status_code=400, detail="mode must be one of: config, auto, pinned")
        if is_v1_model_selection_locked_by_config(config):
            raise HTTPException(
                status_code=409,
                detail="Runtime ingest model selection is locked because detector.model_path is explicitly set in config.",
            )
        if mode == "pinned":
            model_path = _validate_training_base_model_choice(model_path, models_dir)
        else:
            model_path = None

        try:
            save_v1_runtime_model_selection(config, mode=mode, path=model_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return _inference_model_status(config, models_dir)

    @app.post("/api/training/build-dataset")
    async def build_dataset():
        """Trigger dataset build from labeled frames."""
        with _training_lock:
            if _training_state["status"] == "running":
                raise HTTPException(status_code=409, detail="Training or build already in progress")
            _training_state["status"] = "building"
            _training_state["log"] = []
            _training_state["error"] = None

        def _run_build():
            try:
                result = _build_dataset_from_labels(labeling_dir=labeling_dir)
                log_lines = [
                    "Soccer360 Dataset Builder",
                    f"Scanning: {labeling_dir}",
                    f"Output: {result['output_dir']}",
                    f"Val ratio: {result['val_ratio']}",
                    f"Found {result['total_count']} labeled images across {len(result['match_counts'])} matches:",
                ]
                for match_name in sorted(result["match_counts"]):
                    count = result["match_counts"][match_name]
                    log_lines.append(f"  {match_name}: {count} images")
                log_lines.extend([
                    "",
                    f"Dataset built: {result['train_count']} train, {result['val_count']} val",
                    f"YAML: {result['dataset_yaml']}",
                ])
                with _training_lock:
                    _training_state["log"] = log_lines
                    _training_state["status"] = "idle"
                logger.info(
                    "Dataset build completed (%d train, %d val)",
                    result["train_count"],
                    result["val_count"],
                )
            except Exception as exc:
                with _training_lock:
                    _training_state["status"] = "failed"
                    _training_state["error"] = str(exc)
                    if not _training_state["log"]:
                        _training_state["log"] = [str(exc)]
                logger.exception("Dataset build failed")

        threading.Thread(target=_run_build, daemon=True).start()
        return {"ok": True, "status": "building"}

    @app.post("/api/training/train")
    async def start_training(request: Request):
        """Trigger model training."""
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        epochs = body.get("epochs", 50)
        selected_base_model = _validate_training_base_model_choice(body.get("base_model"), models_dir)
        output_model_name = body.get("output_model_name")
        if output_model_name not in (None, ""):
            _validate_flat_name(str(output_model_name), "output model name")
        update_active = bool(body.get("update_active", True))

        with _training_lock:
            if _training_state["status"] in ("running", "building"):
                raise HTTPException(status_code=409, detail="Training or build already in progress")
            _training_state["status"] = "running"
            _training_state["log"] = []
            _training_state["error"] = None

        dataset_yaml = labeling_dir / "dataset" / "dataset.yaml"
        if not dataset_yaml.exists():
            with _training_lock:
                _training_state["status"] = "failed"
                _training_state["error"] = "No dataset found. Build the dataset first."
            raise HTTPException(status_code=400, detail="Dataset not built. Run build-dataset first.")

        def _run_training():
            try:
                result = subprocess.run(
                    [
                        sys.executable, "-m", "src.cli", "train",
                        "--epochs", str(epochs), "--data", str(dataset_yaml),
                        *(["--base-model", selected_base_model] if selected_base_model else []),
                        *(["--output-model-name", str(output_model_name)] if output_model_name else []),
                        *(["--no-update-active"] if not update_active else []),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2hr max
                    cwd=str(APP_ROOT),
                )
                with _training_lock:
                    _training_state["log"] = (
                        result.stdout.splitlines() + result.stderr.splitlines()
                    )[-50:]
                    if result.returncode != 0:
                        _training_state["status"] = "failed"
                        _training_state["error"] = (
                            result.stderr[-500:] if result.stderr else f"Exit code {result.returncode}"
                        )
                    else:
                        _training_state["status"] = "completed"
                logger.info("Training completed (exit=%d)", result.returncode)
            except subprocess.TimeoutExpired:
                with _training_lock:
                    _training_state["status"] = "failed"
                    _training_state["error"] = "Training timed out after 2 hours"
                logger.error("Training timed out")
            except Exception as exc:
                with _training_lock:
                    _training_state["status"] = "failed"
                    _training_state["error"] = str(exc)
                logger.exception("Training failed")

        threading.Thread(target=_run_training, daemon=True).start()
        return {"ok": True, "status": "running", "epochs": epochs}

    # ------------------------------------------------------------------
    # Labeling frame server (for Label Studio to fetch images)
    # ------------------------------------------------------------------

    @app.get("/api/labeling/frames/{match_name}/{filename}")
    async def serve_labeling_frame(match_name: str, filename: str):
        """Serve a hard frame image. Used by Label Studio tasks."""
        if ".." in match_name or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid path")
        frame_path = labeling_dir / match_name / "frames" / filename
        if not frame_path.is_file():
            raise HTTPException(status_code=404, detail="Frame not found")
        return FileResponse(str(frame_path), media_type="image/jpeg")

    # ------------------------------------------------------------------
    # Media / Processed Videos
    # ------------------------------------------------------------------

    @app.get("/api/media/matches")
    async def list_matches():
        """List processed matches with available video files."""
        matches = []
        if processed_dir.is_dir():
            for match_dir in sorted(processed_dir.iterdir(), reverse=True):
                if not match_dir.is_dir() or match_dir.name.startswith("."):
                    continue
                videos = []
                for vf in sorted(match_dir.glob("*.mp4")):
                    videos.append({
                        "name": vf.name,
                        "size_mb": round(vf.stat().st_size / 1e6, 1),
                    })
                # Check for highlights
                hl_dir = highlights_dir / match_dir.name
                if hl_dir.is_dir():
                    for vf in sorted(hl_dir.glob("*.mp4")):
                        videos.append({
                            "name": f"highlights/{vf.name}",
                            "size_mb": round(vf.stat().st_size / 1e6, 1),
                        })
                if videos:
                    # Read metadata if available
                    meta_file = match_dir / "metadata.json"
                    meta = {}
                    if meta_file.exists():
                        try:
                            meta = json.loads(meta_file.read_text())
                        except Exception:
                            pass
                    matches.append({
                        "name": match_dir.name,
                        "canonical_match": meta.get("game_name", match_dir.name),
                        "job_id": meta.get("job_id"),
                        "videos": videos,
                        "mode": meta.get("mode", "--"),
                        "processed_at": meta.get("processed_at", meta.get("processing_start", "--")),
                    })
        return matches

    @app.get("/api/staging/files")
    async def list_staging_files():
        """List eligible video files available in staging."""
        staging_dir.mkdir(parents=True, exist_ok=True)
        return _list_staging_files(
            staging_dir,
            video_extensions=video_extensions,
            ignore_suffixes=ignore_suffixes,
        )

    @app.post("/api/staging/upload")
    async def upload_staging_file(file: UploadFile):
        """Upload a video file into staging for later ingest."""
        filename = Path(file.filename or "").name
        _validate_flat_name(filename, "filename")

        if Path(filename).suffix.lower() in ignore_suffixes:
            raise HTTPException(status_code=400, detail="Invalid staged video file")
        if Path(filename).suffix.lower() not in video_extensions:
            raise HTTPException(status_code=400, detail="Unsupported staged video type")

        staging_dir.mkdir(parents=True, exist_ok=True)
        dest_path = staging_dir / filename
        if dest_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Staging file already exists: {dest_path.name}",
            )

        bytes_written = 0
        try:
            with dest_path.open("xb") as out_f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    bytes_written += len(chunk)
        except FileExistsError:
            raise HTTPException(
                status_code=409,
                detail=f"Staging file already exists: {dest_path.name}",
            )
        except Exception as exc:
            if dest_path.exists():
                dest_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to upload staged file: {exc}")
        finally:
            await file.close()

        if bytes_written <= 0:
            if dest_path.exists():
                dest_path.unlink()
            raise HTTPException(status_code=400, detail="Uploaded file was empty")

        return {
            "ok": True,
            "filename": filename,
            "staging_path": str(dest_path),
            "size_mb": round(bytes_written / 1e6, 1),
        }

    @app.post("/api/staging/import")
    async def import_staging_file(request: Request):
        """Move a selected staging file into ingest for watcher pickup."""
        body = await request.json()
        filename = body.get("filename", "")
        _validate_flat_name(filename, "filename")

        if Path(filename).suffix.lower() in ignore_suffixes:
            raise HTTPException(status_code=400, detail="Invalid staged video file")
        if Path(filename).suffix.lower() not in video_extensions:
            raise HTTPException(status_code=400, detail="Unsupported staged video type")

        source_path = staging_dir / filename
        if not source_path.is_file():
            raise HTTPException(status_code=404, detail="Staged file not found")

        staging_dir.mkdir(parents=True, exist_ok=True)
        ingest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = ingest_dir / filename
        if dest_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Ingest file already exists: {dest_path.name}",
            )

        try:
            _safe_move(source_path, dest_path, overwrite=False)
        except FileExistsError:
            raise HTTPException(
                status_code=409,
                detail=f"Ingest file already exists: {dest_path.name}",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to import staged file: {exc}")

        return {
            "ok": True,
            "filename": filename,
            "source_path": str(source_path),
            "ingest_path": str(dest_path),
        }

    @app.post("/api/media/matches/{match_name}/reset")
    async def reset_match(match_name: str):
        """Delete a processed match family and restore one source video to staging."""
        _validate_flat_name(match_name, "match name")

        selected_dir = processed_dir / match_name
        if not selected_dir.is_dir():
            raise HTTPException(status_code=404, detail="Processed match not found")

        selected_meta = _load_json_file(selected_dir / "metadata.json")
        canonical_match = str(selected_meta.get("game_name") or selected_dir.name)

        active_jobs = store.get_active_jobs_by_input_stem(canonical_match)
        if active_jobs:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot remove '{canonical_match}' while a matching job is queued or running.",
            )

        related: list[tuple[Path, dict]] = [(selected_dir, selected_meta)]
        if processed_dir.is_dir():
            for match_dir in sorted(processed_dir.iterdir()):
                if (
                    match_dir == selected_dir
                    or not match_dir.is_dir()
                    or match_dir.name.startswith(".")
                ):
                    continue

                meta = _load_json_file(match_dir / "metadata.json")
                meta_game_name = meta.get("game_name")
                if isinstance(meta_game_name, str) and meta_game_name == canonical_match:
                    related.append((match_dir, meta))

        job_ids = []
        ingest_paths = []
        restore_candidates = []
        restore_seen = set()
        for match_dir, meta in related:
            job_id = meta.get("job_id")
            if isinstance(job_id, str) and job_id:
                job_ids.append(job_id)

            ingest_source = meta.get("ingest_source_path")
            if isinstance(ingest_source, str) and ingest_source:
                ingest_paths.append(ingest_source)

            for key in (
                "ingest_archived_path",
                "ingest_archive_destination_path",
                "ingest_source_path",
            ):
                candidate = meta.get(key)
                if not isinstance(candidate, str) or not candidate or candidate in restore_seen:
                    continue
                restore_candidates.append(Path(candidate))
                restore_seen.add(candidate)

        job_ids = sorted(dict.fromkeys(job_ids))
        ingest_paths = list(dict.fromkeys(ingest_paths))

        warnings = []
        deleted_processed_count = 0
        deleted_highlights_count = 0
        for match_dir, _ in related:
            highlights_match_dir = highlights_dir / match_dir.name
            if highlights_match_dir.exists():
                shutil.rmtree(highlights_match_dir)
                deleted_highlights_count += 1
            if match_dir.exists():
                shutil.rmtree(match_dir)
                deleted_processed_count += 1

        match_labeling_dir = labeling_dir / canonical_match
        labeling_deleted = False
        if match_labeling_dir.exists():
            shutil.rmtree(match_labeling_dir)
            labeling_deleted = True

        dataset_dir = labeling_dir / "dataset"
        dataset_invalidated = False
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            dataset_invalidated = True

        purged_job_count = store.delete_jobs(job_ids)

        dedupe_entries_removed = 0
        try:
            dedupe_store = ProcessedIngestStore(processed_state_path)
            dedupe_entries_removed = dedupe_store.delete_paths(ingest_paths)
        except Exception as exc:
            warnings.append(f"Failed to update watcher dedupe state: {exc}")

        staging_dir.mkdir(parents=True, exist_ok=True)
        restored_staging_path = None
        found_existing_candidate = False
        for candidate in restore_candidates:
            if not candidate.exists():
                continue

            found_existing_candidate = True
            suffix = "".join(candidate.suffixes)
            dest_path = _unique_file_path(
                staging_dir,
                f"{canonical_match}_reprocess{suffix}",
            )
            try:
                _safe_move(candidate, dest_path, overwrite=False)
                restored_staging_path = str(dest_path)
                break
            except Exception as exc:
                warnings.append(
                    f"Failed to restore source video from {candidate}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if restored_staging_path is None and not found_existing_candidate:
            warnings.append("No restorable source video found in archive or ingest.")

        return {
            "ok": True,
            "canonical_match": canonical_match,
            "deleted_processed_dirs_count": deleted_processed_count,
            "deleted_highlights_dirs_count": deleted_highlights_count,
            "labeling_deleted": labeling_deleted,
            "dataset_invalidated": dataset_invalidated,
            "purged_job_ids": job_ids,
            "purged_job_count": purged_job_count,
            "dedupe_entries_removed": dedupe_entries_removed,
            "restored_staging_path": restored_staging_path,
            "warnings": warnings,
        }

    @app.get("/api/media/{match_name}/{filename:path}")
    async def stream_video(match_name: str, filename: str):
        """Serve a processed video file with range request support."""
        # Sanitize path components
        if ".." in match_name or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid path")

        if filename.startswith("highlights/"):
            clip_name = filename[len("highlights/"):]
            video_path = highlights_dir / match_name / clip_name
        else:
            video_path = processed_dir / match_name / filename

        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Video not found")

        return FileResponse(
            str(video_path),
            media_type="video/mp4",
            filename=filename,
        )

    # ------------------------------------------------------------------
    # SSE event stream
    # ------------------------------------------------------------------

    @app.get("/api/events")
    async def event_stream(request: Request, after: int = 0):
        """SSE endpoint that streams new phase events and periodic GPU snapshots."""

        async def generate():
            last_id = after
            gpu_counter = 0

            while True:
                if await request.is_disconnected():
                    break

                # Stream new phase events
                events = store.get_events_since(after_id=last_id)
                for ev in events:
                    last_id = ev["id"]
                    yield {
                        "event": f"phase_{ev['status']}",
                        "id": str(last_id),
                        "data": json.dumps(ev),
                    }

                # Stream pending decisions
                decisions = store.get_pending_decisions()
                for d in decisions:
                    yield {
                        "event": "decision_pending",
                        "data": json.dumps(d),
                    }

                # Periodic hardware snapshots (every ~5 seconds = 5 iterations)
                gpu_counter += 1
                if gpu_counter >= 5:
                    gpu_counter = 0
                    snap = gpu_utilization_snapshot()
                    if snap:
                        yield {
                            "event": "gpu_snapshot",
                            "data": json.dumps(snap),
                        }
                    sys_snap = cpu_ram_snapshot()
                    if sys_snap:
                        yield {
                            "event": "system_snapshot",
                            "data": json.dumps(sys_snap),
                        }

                # Periodic status heartbeat
                status = store.get_current_status()
                yield {
                    "event": "status",
                    "data": json.dumps(status),
                }

                await asyncio.sleep(1)

        return EventSourceResponse(generate())

    return app
