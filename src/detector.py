"""YOLO-based ball detection with streaming ffmpeg frame input."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .utils import FFmpegFrameReader, VideoMeta, write_detections_jsonl

logger = logging.getLogger("soccer360.detector")


class Detector:
    """Streaming ball detector using Ultralytics YOLO.

    Reads frames from ffmpeg pipe in batches, runs GPU inference,
    and writes per-frame detections to a JSONL file.
    Frames are never stored on disk.
    """

    def __init__(self, config: dict):
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

        self.device = "cuda:0"
        self._model = None

    def _load_model(self):
        from ultralytics import YOLO

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
    ):
        """Stream frames from video, run batched detection, write JSONL.

        If process_every_n_frames > 1, only every Nth frame is sent through
        YOLO. Skipped frames get detections interpolated from neighbors.
        """
        if self._model is None:
            self._load_model()

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

        logger.info(
            "Detection complete: %d frames processed, %d detected, %d total detections",
            frame_count, len(detected_detections), len(all_detections),
        )
        write_detections_jsonl(all_detections, output_path)

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

    def _detect_batch(
        self, frames: list[np.ndarray], indices: list[int]
    ) -> list[dict]:
        """Run YOLO inference on a batch of frames."""
        if self.tiling_enabled:
            return self._detect_batch_tiled(frames, indices)

        results = self._model.predict(
            frames,
            device=self.device,
            conf=self.confidence_threshold,
            iou=self.nms_iou,
            verbose=False,
        )

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
