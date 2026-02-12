"""YOLO model training and hard-frame export for active learning."""

from __future__ import annotations

import logging
from pathlib import Path

from .utils import VideoMeta, extract_frame, load_detections_jsonl, probe_video

logger = logging.getLogger("soccer360.trainer")


class Trainer:
    """Fine-tune YOLO ball detection model and export hard frames."""

    def __init__(self, config: dict):
        model_cfg = config.get("model", {})
        self.base_model = model_cfg.get("base_model", "yolov8s.pt")
        self.model_dir = Path(config["paths"].get("models", "/tank/models"))

    def run(self, data: str, epochs: int = 50):
        """Fine-tune YOLO model on labeled ball dataset.

        Args:
            data: Path to YOLO dataset YAML file.
            epochs: Number of training epochs.
        """
        from ultralytics import YOLO

        version = self._next_version()
        run_name = f"ball_model_{version}"

        logger.info("Starting training: %s (base=%s, epochs=%d)", run_name, self.base_model, epochs)

        model = YOLO(self.base_model)
        results = model.train(
            data=data,
            epochs=epochs,
            imgsz=640,
            batch=16,
            device="cuda:0",
            project=str(self.model_dir),
            name=run_name,
            exist_ok=False,
            patience=10,
        )

        # Copy best weights to a standard location
        best_path = self.model_dir / run_name / "weights" / "best.pt"
        if best_path.exists():
            latest = self.model_dir / "ball_best.pt"
            import shutil
            shutil.copy2(str(best_path), str(latest))
            logger.info("Best model saved: %s (copied to %s)", best_path, latest)

        logger.info("Training complete: %s", run_name)
        return results

    def export_tensorrt(self, model_path: str | Path, int8: bool = True):
        """Export YOLO model to TensorRT engine.

        INT8 quantization is optimal for Tesla P40 (47 TOPS INT8).
        """
        from ultralytics import YOLO

        logger.info("Exporting to TensorRT (INT8=%s): %s", int8, model_path)
        model = YOLO(str(model_path))
        engine_path = model.export(
            format="engine",
            int8=int8,
            dynamic=False,
            simplify=True,
            workspace=8,
        )
        logger.info("TensorRT engine exported: %s", engine_path)
        return engine_path

    def export_hard_frames(
        self,
        video_path: str | Path,
        detections_path: Path,
        threshold: float = 0.3,
        output_dir: Path = Path("/tank/labeling"),
    ):
        """Export frames where detection confidence is below threshold.

        These "hard frames" are candidates for manual labeling to improve
        the model through active learning.

        Exports:
          - Frames with detections below confidence threshold
          - Frames with NO detections at all (ball completely lost)
        """
        video_path = Path(video_path)
        meta = probe_video(video_path)
        detections = load_detections_jsonl(detections_path)

        logger.info(
            "Exporting hard frames: threshold=%.2f, video=%s, %d detections",
            threshold, video_path.name, len(detections),
        )

        # Find frames with low-confidence detections
        hard_frames: set[int] = set()
        frames_with_dets: set[int] = set()

        for det in detections:
            frames_with_dets.add(det["frame"])
            if det["confidence"] < threshold:
                hard_frames.add(det["frame"])

        # Also include frames with no detections at all
        max_frame = max(frames_with_dets) if frames_with_dets else 0
        for f in range(max_frame + 1):
            if f not in frames_with_dets:
                hard_frames.add(f)

        logger.info(
            "Found %d hard frames (%d low-conf, %d no-detection)",
            len(hard_frames),
            sum(1 for f in hard_frames if f in frames_with_dets),
            sum(1 for f in hard_frames if f not in frames_with_dets),
        )

        # Limit to a reasonable number (sample if too many)
        max_export = 500
        if len(hard_frames) > max_export:
            import random
            hard_frames = set(random.sample(sorted(hard_frames), max_export))
            logger.info("Sampled down to %d frames", max_export)

        # Export frames
        game_dir = output_dir / video_path.stem
        images_dir = game_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        exported = 0
        for frame_idx in sorted(hard_frames):
            out_path = images_dir / f"frame_{frame_idx:06d}.jpg"
            try:
                extract_frame(video_path, frame_idx, meta.fps, out_path)
                exported += 1
            except Exception:
                logger.warning("Failed to extract frame %d", frame_idx)

        logger.info(
            "Exported %d hard frames to %s",
            exported, images_dir,
        )

    def _next_version(self) -> str:
        """Generate a timestamp-based version string.

        Respects SOCCER360_RUN_NAME env var if set (used by train_ball.sh).
        """
        import os
        from datetime import datetime

        env_name = os.environ.get("SOCCER360_RUN_NAME")
        if env_name:
            return env_name.replace("ball_model_", "")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        return datetime.now().strftime("%Y%m%d_%H%M")
