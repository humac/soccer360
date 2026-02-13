"""YOLO-based ball detection with streaming ffmpeg frame input."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from .utils import (
    FFmpegFrameReader,
    VideoMeta,
    pixel_to_yaw_pitch,
    wrap_angle_deg,
    write_detections_jsonl,
    write_json,
)

logger = logging.getLogger("soccer360.detector")
DEFAULT_V1_MODEL_PATH = "/app/yolov8s.pt"
DEFAULT_V1_MODEL_ALIASES = {DEFAULT_V1_MODEL_PATH, "yolov8s.pt"}


def resolve_model_path(config: dict) -> tuple[str | None, str]:
    """Resolve the best available ball-detection model path.

    Priority:
      1. {paths.models}/ball_best.pt  (fine-tuned model on /tank)
      2. config["model"]["path"]      (e.g. /app/models/ball_best.pt)
      3. /app/models/ball_base.pt     (repo baseline; copied to tank on first use)
      4. None -> NO_DETECT mode

    Returns:
        (resolved_path or None, "normal" or "no_detect")
    """
    import shutil

    tank_model = Path(config.get("paths", {}).get("models", "/tank/models")) / "ball_best.pt"
    config_model = Path(config.get("model", {}).get("path", "yolov8s.pt"))
    base_model = Path("/app/models/ball_base.pt")

    if tank_model.exists():
        logger.info("Model resolved: %s (fine-tuned)", tank_model)
        return str(tank_model), "normal"

    if config_model.exists():
        logger.info("Model resolved: %s (config path)", config_model)
        return str(config_model), "normal"

    if base_model.exists():
        tank_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(base_model), str(tank_model))
        logger.info(
            "Model resolved: %s (base model, copied to %s)", base_model, tank_model
        )
        return str(tank_model), "normal"

    logger.warning("No ball detection model found -- entering NO_DETECT mode")
    return None, "no_detect"


def resolve_model_path_v1(
    config: dict,
    models_dir: str = "/app/models",
    base_model_path: str = DEFAULT_V1_MODEL_PATH,
) -> tuple[str | None, str]:
    """Resolve model for V1 bootstrap detection.

    Priority (when detection.path is the default "yolov8s.pt"):
      1. {models_dir}/ball_best.pt  (fine-tuned model)
      2. {base_model_path}          (baked canonical yolov8s.pt)
      3. None -> NO_DETECT (if mode.allow_no_model)
      4. RuntimeError (if allow_no_model is false)

    When detection.path is explicitly overridden to a non-default value,
    that path is tried first (absolute, then relative to /app).
    """
    model_path, model_source = resolve_v1_model_path_and_source(
        config, models_dir=models_dir, base_model_path=base_model_path
    )
    config["_v1_model_resolution"] = {"path": model_path, "source": model_source}

    if model_path is None:
        logger.warning("No model found -- entering NO_DETECT mode")
        return None, "no_detect"
    return model_path, "normal"


def _normalize_model_path(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_runtime_path(path_value: str) -> str:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(Path("/app") / path_obj)


def _v1_default_aliases(base_model_path: str) -> set[str]:
    aliases = set(DEFAULT_V1_MODEL_ALIASES)
    normalized = _normalize_model_path(base_model_path)
    if normalized:
        aliases.add(normalized)
        aliases.add(Path(normalized).name)
    return aliases


def _configured_v1_model_candidate(config: dict) -> tuple[str, str]:
    """Return configured V1 model candidate path and its source enum."""
    det_cfg = config.get("detector", {})
    v1_cfg = config.get("detection", {})

    detector_model_path = _normalize_model_path(det_cfg.get("model_path"))
    legacy_detection_path = _normalize_model_path(v1_cfg.get("path"))

    if detector_model_path:
        return detector_model_path, "detector.model_path"
    if legacy_detection_path:
        return legacy_detection_path, "detection.path"
    return DEFAULT_V1_MODEL_PATH, "default"


def resolve_v1_model_path_and_source(
    config: dict,
    models_dir: str = "/app/models",
    base_model_path: str = DEFAULT_V1_MODEL_PATH,
) -> tuple[str | None, str]:
    """Resolve effective V1 model path and source enum for runtime/verifier.

    Source enum is stable and limited to:
      - detector.model_path
      - detection.path
      - default

    Resolution precedence is configuration-first:
      detector.model_path > detection.path > default

    Behavior:
      - Non-default detector.model_path is an explicit override.
      - Missing detector.model_path, or detector.model_path set to default path,
        preserves fine-tuned preference behavior.
      - Legacy detection.path explicit values are honored when present and valid.
    """
    allow_no_model = config.get("mode", {}).get("allow_no_model", False)
    fine_tuned = Path(models_dir) / "ball_best.pt"
    base_model = Path(base_model_path)
    default_aliases = _v1_default_aliases(base_model_path)

    configured_path, configured_source = _configured_v1_model_candidate(config)
    candidate_path = _resolve_runtime_path(configured_path)

    # Explicit detector.model_path override is only for non-default values.
    if (
        configured_source == "detector.model_path"
        and configured_path not in default_aliases
    ):
        return candidate_path, "detector.model_path"

    # Legacy explicit detection.path override remains supported.
    if (
        configured_source == "detection.path"
        and configured_path not in default_aliases
    ):
        if Path(candidate_path).exists():
            return candidate_path, "detection.path"
        logger.warning(
            "Explicit model path %s not found, falling back to default resolution",
            candidate_path,
        )

    # Default/fallback resolution preserves existing fine-tuned preference.
    if fine_tuned.exists():
        return str(fine_tuned), "default"
    if base_model.exists():
        return str(base_model), "default"
    if allow_no_model:
        return None, "default"

    raise RuntimeError(
        f"No ball detection model found (checked {fine_tuned}, {base_model}) "
        "and mode.allow_no_model is false"
    )


class Detector:
    """Streaming ball detector using Ultralytics YOLO.

    Reads frames from ffmpeg pipe in batches, runs GPU inference,
    and writes per-frame detections to a JSONL file.
    Frames are never stored on disk.
    """

    def __init__(self, config: dict):
        self._v1_mode = "detection" in config

        if self._v1_mode:
            v1_cfg = config["detection"]
            filt_cfg = config.get("filters", {})
            resolution_meta = config.get("_v1_model_resolution", {})
            resolved_model_path = _normalize_model_path(resolution_meta.get("path"))
            resolved_model_source = _normalize_model_path(resolution_meta.get("source"))
            if not resolved_model_path:
                resolved_model_path, resolved_model_source = resolve_v1_model_path_and_source(
                    config
                )
            self.model_path = resolved_model_path
            self.model_source = resolved_model_source or "default"
            self.device = v1_cfg.get("device", "cuda:0")
            img_size = v1_cfg.get("img_size", 960)
            self.det_resolution = [img_size * 2, img_size]
            self.confidence_threshold = v1_cfg.get("conf", 0.35)
            self.nms_iou = v1_cfg.get("iou", 0.5)
            self.classes = v1_cfg.get("classes", [32])
            self.max_det = v1_cfg.get("max_det", 20)
            self.half = v1_cfg.get("half", True)
            self.batch_size = 1
            self.batch_size_trt = 1
            self.process_every_n = 1
            self.tiling_enabled = False
            self.backend = "fp32"
            self.tensorrt_path = None
            # Y-range filter
            self.min_y_frac = filt_cfg.get("min_y_frac", 0.20)
            self.max_y_frac = filt_cfg.get("max_y_frac", 0.98)
        else:
            # Legacy init (existing code unchanged)
            model_cfg = config.get("model", {})
            det_cfg = config.get("detector", {})

            self.model_path = model_cfg.get("path", "yolov8s.pt")
            self.backend = model_cfg.get("backend", "fp32")
            self.tensorrt_path = model_cfg.get("tensorrt_path")

            self.batch_size = det_cfg.get("batch_size", 16)
            self.batch_size_trt = det_cfg.get("batch_size_tensorrt", 32)
            self.det_resolution = det_cfg.get("resolution", [1920, 960])
            self.confidence_threshold = det_cfg.get("confidence_threshold", 0.25)
            self.nms_iou = det_cfg.get("nms_iou_threshold", 0.45)
            self.process_every_n = det_cfg.get("process_every_n_frames", 1)

            self.tiling = det_cfg.get("tiling", {})
            self.tiling_enabled = self.tiling.get("enabled", False)
            self.tile_count = self.tiling.get("tiles", 4)  # 2x2
            self.tile_overlap = self.tiling.get("overlap", 0.1)
            self.model_source = "model.path"

        # Field-of-Interest filtering
        foi_cfg = config.get("field_of_interest", {})
        self.foi_enabled = foi_cfg.get("enabled", False)
        self.foi_center_mode = foi_cfg.get("center_mode", "fixed")
        self.foi_center_yaw = foi_cfg.get("center_yaw_deg", 0.0)
        self.foi_yaw_window = foi_cfg.get("yaw_window_deg", 200.0)
        self.foi_pitch_min = foi_cfg.get("pitch_min_deg", -45.0)
        self.foi_pitch_max = foi_cfg.get("pitch_max_deg", 20.0)
        self.foi_auto_sample_sec = foi_cfg.get("auto_sample_seconds", 30.0)
        self.foi_auto_min_conf = foi_cfg.get("auto_min_conf", 0.25)
        self._effective_center_yaw: float | None = None

        self.device = getattr(self, "device", "cuda:0")
        self._model = None

    def _load_model(self):
        from ultralytics import YOLO

        # V1: auto-disable half precision if no CUDA
        if self._v1_mode and self.half:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.info("CUDA not available, disabling FP16")
                    self.half = False
            except ImportError:
                self.half = False

        use_trt = self.backend == "tensorrt_int8"
        path = self.tensorrt_path if use_trt else self.model_path
        logger.info("Loading model from %s (backend=%s)", path, self.backend)
        self._model = YOLO(path)

        # Warmup inference
        dummy = np.zeros(
            (self.det_resolution[1], self.det_resolution[0], 3), dtype=np.uint8
        )
        self._model.predict(dummy, device=self.device, verbose=False)
        logger.info("Model loaded and warmed up")

    @property
    def _effective_batch_size(self) -> int:
        return self.batch_size_trt if self.backend == "tensorrt_int8" else self.batch_size

    def run_streaming(
        self, video_path: str | Path, meta: VideoMeta, output_path: Path
    ) -> int:
        """Stream frames from video, run batched detection, write JSONL.

        If process_every_n_frames > 1, only every Nth frame is sent through
        YOLO. Skipped frames get detections interpolated from neighbors.
        """
        if self._model is None:
            self._load_model()

        # Recompute FoI auto-center per run (do not leak prior run state).
        self._effective_center_yaw = None
        logger.info("Model resolved: %s (source=%s)", self.model_path, self.model_source)

        det_w, det_h = self.det_resolution
        bs = self._effective_batch_size
        skip_n = self.process_every_n

        logger.info(
            "Starting detection: %s (%d frames, %dx%d det resolution, batch=%d, skip=%d)",
            video_path, meta.total_frames, det_w, det_h, bs, skip_n,
        )

        reader = FFmpegFrameReader(
            video_path,
            output_width=det_w,
            output_height=det_h,
            fps=meta.fps,
        )

        detected_detections: list[dict] = []
        batch_frames: list[np.ndarray] = []
        batch_indices: list[int] = []
        frame_count = 0

        for frame_idx, frame in enumerate(reader):
            # Skip frames if configured
            if skip_n > 1 and frame_idx % skip_n != 0:
                frame_count = frame_idx + 1
                continue

            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            if len(batch_frames) == bs:
                dets = self._detect_batch(batch_frames, batch_indices)
                detected_detections.extend(dets)
                batch_frames.clear()
                batch_indices.clear()

            frame_count = frame_idx + 1
            if frame_count % 1000 == 0:
                logger.info("Detected %d / %d frames", frame_count, meta.total_frames)

        # Process remaining
        if batch_frames:
            dets = self._detect_batch(batch_frames, batch_indices)
            detected_detections.extend(dets)

        # Interpolate skipped frames
        if skip_n > 1:
            all_detections = self._interpolate_skipped(
                detected_detections, frame_count, skip_n
            )
        else:
            all_detections = detected_detections

        # Field-of-Interest filtering
        all_detections = self._filter_foi(
            all_detections, det_w, det_h, meta.fps, output_path.parent,
        )

        # V1: y-range filter + best-per-frame selection
        if self._v1_mode:
            all_detections = self._filter_y_range(
                all_detections, det_h, self.min_y_frac, self.max_y_frac
            )
            all_detections = self._select_best_per_frame(all_detections)
            # Rewrite to V1 canonical field names
            v1_detections = []
            for det in all_detections:
                frame = det.get("frame", det.get("frame_index", 0))
                v1_detections.append({
                    "frame_index": frame,
                    "time_sec": round(frame / meta.fps, 4) if meta.fps > 0 else 0.0,
                    "bbox_xyxy": det.get("bbox", det.get("bbox_xyxy")),
                    "conf": det.get("confidence", det.get("conf", 0.0)),
                    "class_id": det.get("class", det.get("class_id", 32)),
                })
            all_detections = v1_detections

        logger.info(
            "Detection complete: %d frames processed, %d detected, %d total detections",
            frame_count, len(detected_detections), len(all_detections),
        )
        write_detections_jsonl(all_detections, output_path)
        return frame_count

    @staticmethod
    def _interpolate_skipped(
        detections: list[dict], total_frames: int, skip_n: int
    ) -> list[dict]:
        """Interpolate detections for skipped frames from neighbors."""
        # Group detections by frame
        by_frame: dict[int, list[dict]] = {}
        for d in detections:
            by_frame.setdefault(d["frame"], []).append(d)

        detected_frames = sorted(by_frame.keys())
        all_detections = list(detections)  # start with originals

        for i in range(len(detected_frames) - 1):
            f_start = detected_frames[i]
            f_end = detected_frames[i + 1]

            if f_end - f_start <= 1:
                continue
            if f_end - f_start > skip_n:
                # Do not bridge true detector gaps; only fill skip-induced holes.
                continue

            dets_start = by_frame[f_start]
            dets_end = by_frame[f_end]

            if not dets_start or not dets_end:
                continue

            # Interpolate the highest-confidence detection from each side
            best_start = max(dets_start, key=lambda d: d["confidence"])
            best_end = max(dets_end, key=lambda d: d["confidence"])

            for gap_frame in range(f_start + 1, f_end):
                t = (gap_frame - f_start) / (f_end - f_start)
                interp_bbox = [
                    best_start["bbox"][j] + t * (best_end["bbox"][j] - best_start["bbox"][j])
                    for j in range(4)
                ]
                interp_conf = best_start["confidence"] + t * (
                    best_end["confidence"] - best_start["confidence"]
                )
                all_detections.append({
                    "frame": gap_frame,
                    "bbox": interp_bbox,
                    "confidence": interp_conf * 0.8,  # discount interpolated
                    "class": 0,
                    "interpolated": True,
                })

        all_detections.sort(key=lambda d: d["frame"])
        return all_detections

    def _filter_foi(
        self,
        detections: list[dict],
        width: int,
        height: int,
        fps: float,
        work_dir: Path,
    ) -> list[dict]:
        """Filter detections outside the configured Field-of-Interest region.

        Writes foi_meta.json to work_dir with effective center and stats.
        """
        if not self.foi_enabled:
            return detections

        # Determine effective center yaw (compute once per run)
        if self._effective_center_yaw is None:
            fallback = False
            sample_count = 0

            if self.foi_center_mode == "auto":
                self._effective_center_yaw, sample_count, fallback = (
                    self._compute_auto_center(detections, width, height, fps)
                )
            else:
                self._effective_center_yaw = self.foi_center_yaw

            # Write foi_meta.json immediately
            foi_meta = {
                "enabled": True,
                "center_mode": self.foi_center_mode,
                "effective_center_yaw_deg": self._effective_center_yaw,
                "configured_center_yaw_deg": self.foi_center_yaw,
                "yaw_window_deg": self.foi_yaw_window,
                "pitch_min_deg": self.foi_pitch_min,
                "pitch_max_deg": self.foi_pitch_max,
                "sample_count": sample_count,
                "fallback": fallback,
            }
            write_json(foi_meta, work_dir / "foi_meta.json")

        # Filter detections
        half_window = self.foi_yaw_window / 2.0
        center = self._effective_center_yaw
        total = len(detections)

        kept = []
        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            yaw, pitch = pixel_to_yaw_pitch(cx, cy, width, height)

            if abs(wrap_angle_deg(yaw - center)) > half_window:
                continue
            if pitch < self.foi_pitch_min or pitch > self.foi_pitch_max:
                continue
            kept.append(det)

        filtered_pct = (1.0 - len(kept) / total) * 100.0 if total > 0 else 0.0
        logger.info(
            "FoI filter: %d total -> %d kept (%.1f%% filtered), effective center_yaw=%.1f",
            total, len(kept), filtered_pct, center,
        )
        return kept

    def _compute_auto_center(
        self,
        detections: list[dict],
        width: int,
        height: int,
        fps: float,
    ) -> tuple[float, int, bool]:
        """Compute auto-center yaw from histogram peak of early detections.

        Returns (effective_center_yaw, sample_count, is_fallback).
        """
        max_frame = int(fps * self.foi_auto_sample_sec)

        # Collect samples: early frames, sufficient confidence, valid pitch
        sample_yaws = []
        for det in detections:
            if det["frame"] >= max_frame:
                continue
            if det.get("confidence", 0) < self.foi_auto_min_conf:
                continue

            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            yaw, pitch = pixel_to_yaw_pitch(cx, cy, width, height)

            if pitch < self.foi_pitch_min or pitch > self.foi_pitch_max:
                continue
            sample_yaws.append(yaw)

        sample_count = len(sample_yaws)

        if sample_count == 0:
            logger.warning(
                "FoI auto-center: no samples passed filters, "
                "falling back to center_yaw_deg=%.1f",
                self.foi_center_yaw,
            )
            return self.foi_center_yaw, 0, True

        # Build 5-degree histogram over (-180, 180]
        bin_width = 5.0
        num_bins = 72  # 360 / 5
        bins = [0] * num_bins

        for yaw in sample_yaws:
            idx = int((wrap_angle_deg(yaw) + 180.0) / bin_width) % num_bins
            bins[idx] += 1

        # Find peak bin
        peak_idx = max(range(num_bins), key=lambda i: bins[i])
        peak_center = -180.0 + (peak_idx + 0.5) * bin_width

        # Collect yaws in peak bin ± 1 neighbor (with wraparound)
        neighbor_indices = {
            (peak_idx - 1) % num_bins,
            peak_idx,
            (peak_idx + 1) % num_bins,
        }
        local_yaws = []
        for yaw in sample_yaws:
            idx = int((wrap_angle_deg(yaw) + 180.0) / bin_width) % num_bins
            if idx in neighbor_indices:
                local_yaws.append(yaw)

        # Circular mean of local cluster
        sin_sum = sum(math.sin(math.radians(y)) for y in local_yaws)
        cos_sum = sum(math.cos(math.radians(y)) for y in local_yaws)
        effective = math.degrees(math.atan2(sin_sum, cos_sum))

        logger.info(
            "FoI auto-center: %d samples, peak_bin=%.0f, effective center_yaw=%.1f",
            sample_count, peak_center, effective,
        )
        return effective, sample_count, False

    def _detect_batch(
        self, frames: list[np.ndarray], indices: list[int]
    ) -> list[dict]:
        """Run YOLO inference on a batch of frames."""
        if self.tiling_enabled:
            return self._detect_batch_tiled(frames, indices)

        predict_kwargs = {
            "device": self.device,
            "conf": self.confidence_threshold,
            "iou": self.nms_iou,
            "verbose": False,
        }
        if self._v1_mode:
            predict_kwargs["classes"] = self.classes
            predict_kwargs["max_det"] = self.max_det
            predict_kwargs["half"] = self.half
        results = self._model.predict(frames, **predict_kwargs)

        detections = []
        for idx, result in zip(indices, results):
            for box in result.boxes:
                detections.append({
                    "frame": idx,
                    "bbox": box.xyxy[0].cpu().tolist(),
                    "confidence": float(box.conf[0]),
                    "class": int(box.cls[0]),
                })
        return detections

    def _detect_batch_tiled(
        self, frames: list[np.ndarray], indices: list[int]
    ) -> list[dict]:
        """Run tiled detection for better small-ball recall.

        Splits each frame into overlapping tiles, runs YOLO on each tile,
        and merges results with NMS.
        """
        detections = []
        for frame, idx in zip(frames, indices):
            frame_dets = self._detect_frame_tiled(frame, idx)
            detections.extend(frame_dets)
        return detections

    def _detect_frame_tiled(self, frame: np.ndarray, frame_idx: int) -> list[dict]:
        """Detect on a single frame using 2x2 tiling."""
        h, w = frame.shape[:2]
        rows, cols = 2, 2
        tile_h = h // rows
        tile_w = w // cols
        overlap_h = int(tile_h * self.tile_overlap)
        overlap_w = int(tile_w * self.tile_overlap)

        all_boxes = []
        all_confs = []

        for r in range(rows):
            for c in range(cols):
                y1 = max(0, r * tile_h - overlap_h)
                y2 = min(h, (r + 1) * tile_h + overlap_h)
                x1 = max(0, c * tile_w - overlap_w)
                x2 = min(w, (c + 1) * tile_w + overlap_w)

                tile = frame[y1:y2, x1:x2]
                results = self._model.predict(
                    tile,
                    device=self.device,
                    conf=self.confidence_threshold,
                    iou=self.nms_iou,
                    verbose=False,
                )

                for box in results[0].boxes:
                    bx = box.xyxy[0].cpu().numpy()
                    # Map tile coords back to full frame
                    bx[0] += x1
                    bx[1] += y1
                    bx[2] += x1
                    bx[3] += y1
                    all_boxes.append(bx)
                    all_confs.append(float(box.conf[0]))

        # NMS across tiles
        if not all_boxes:
            return []

        boxes_arr = np.array(all_boxes)
        confs_arr = np.array(all_confs)
        keep = self._nms(boxes_arr, confs_arr, self.nms_iou)

        detections = []
        for i in keep:
            detections.append({
                "frame": frame_idx,
                "bbox": boxes_arr[i].tolist(),
                "confidence": confs_arr[i],
                "class": 0,
            })
        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """Simple NMS implementation."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(int(i))

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep

    @staticmethod
    def _filter_y_range(
        detections: list[dict], height: int, min_y_frac: float, max_y_frac: float
    ) -> list[dict]:
        """Reject detections outside vertical band [min_y_frac, max_y_frac]."""
        kept = []
        for det in detections:
            bbox = det.get("bbox", det.get("bbox_xyxy", [0, 0, 0, 0]))
            cy = (bbox[1] + bbox[3]) / 2.0
            y_frac = cy / height if height > 0 else 0.0
            if min_y_frac <= y_frac <= max_y_frac:
                kept.append(det)
        return kept

    @staticmethod
    def _select_best_per_frame(detections: list[dict]) -> list[dict]:
        """Keep only highest-confidence detection per frame."""
        by_frame: dict[int, dict] = {}
        for det in detections:
            frame = det.get("frame", det.get("frame_index", -1))
            conf = det.get("confidence", det.get("conf", 0.0))
            if frame not in by_frame or conf > by_frame[frame].get(
                "confidence", by_frame[frame].get("conf", 0.0)
            ):
                by_frame[frame] = det
        return sorted(
            by_frame.values(), key=lambda d: d.get("frame", d.get("frame_index", 0))
        )
