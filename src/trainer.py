"""YOLO model training and hard-frame export for active learning."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("soccer360.trainer")


class Trainer:
    """Fine-tune YOLO ball detection model and export hard frames."""

    def __init__(self, config: dict):
        self.config = config
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})
        self.base_model = model_cfg.get("base_model", "yolo26l.pt")
        self.model_dir = Path(config["paths"].get("models", "/tank/models"))
        self.train_workers = int(training_cfg.get("workers", 0))

    def run(
        self,
        data: str,
        epochs: int = 50,
        base_model: str | Path | None = None,
        output_model_name: str | None = None,
        update_active: bool = True,
    ):
        """Fine-tune YOLO model on labeled ball dataset.

        Args:
            data: Path to YOLO dataset YAML file.
            epochs: Number of training epochs.
        """
        from ultralytics import YOLO

        version = self._next_version()
        run_name = f"ball_model_{version}"
        resolved_base_model = self._resolve_training_base_model(base_model)

        logger.info(
            "Starting training: %s (base=%s, epochs=%d)",
            run_name,
            resolved_base_model,
            epochs,
        )

        model = YOLO(str(resolved_base_model))
        results = model.train(
            data=data,
            epochs=epochs,
            imgsz=640,
            batch=16,
            device="cuda:0",
            workers=self.train_workers,
            project=str(self.model_dir),
            name=run_name,
            exist_ok=False,
            patience=10,
        )

        # Copy best weights to the requested model slot(s)
        best_path = self.model_dir / run_name / "weights" / "best.pt"
        if best_path.exists():
            import shutil
            target_name = self._resolve_output_model_name(output_model_name)
            target_path = self.model_dir / target_name
            shutil.copy2(str(best_path), str(target_path))

            if update_active and target_name != "ball_best.pt":
                active_path = self.model_dir / "ball_best.pt"
                shutil.copy2(str(best_path), str(active_path))
                logger.info(
                    "Best model saved: %s (copied to %s and active %s)",
                    best_path,
                    target_path,
                    active_path,
                )
            else:
                logger.info("Best model saved: %s (copied to %s)", best_path, target_path)

            self._cleanup_run_weight_artifacts(run_name)

        logger.info("Training complete: %s", run_name)
        return results

    def _resolve_training_base_model(self, override: str | Path | None = None) -> Path:
        """Resolve a local base model path for offline-safe training."""
        configured_value = override if override is not None else self.base_model
        configured = Path(str(configured_value))
        candidate_paths: list[Path] = []

        if configured.is_absolute():
            candidate_paths.append(configured)
        else:
            candidate_paths.append(Path.cwd() / configured)
            candidate_paths.append(Path("/app") / configured)

        active_model = self.model_dir / "ball_best.pt"
        baked_default = Path("/app/models/yolo26l.pt")

        for candidate in candidate_paths:
            if candidate.is_file():
                return candidate

        if active_model.is_file():
            logger.info(
                "Configured training base model '%s' is not available locally; "
                "falling back to active model %s",
                configured_value,
                active_model,
            )
            return active_model

        if baked_default.is_file():
            logger.info(
                "Configured training base model '%s' is not available locally; "
                "falling back to baked default %s",
                configured_value,
                baked_default,
            )
            return baked_default

        raise RuntimeError(
            "Training base model is not available locally. "
            f"Configured base_model={configured_value!r}. "
            "Set model.base_model to a local .pt file or ensure /app/models/yolo26l.pt exists."
        )

    def _resolve_output_model_name(self, override: str | None = None) -> str:
        """Resolve a safe model filename for the promoted checkpoint."""
        name = (override or "ball_best").strip()
        if (
            not name
            or name in {".", ".."}
            or name.startswith(".")
            or "/" in name
            or "\\" in name
            or ".." in name
            or Path(name).name != name
        ):
            raise ValueError(f"Invalid output model name: {override!r}")
        if not name.lower().endswith(".pt"):
            name += ".pt"
        return name

    def _cleanup_run_weight_artifacts(self, run_name: str) -> None:
        """Delete per-run Ultralytics weight artifacts after successful promotion."""
        weights_dir = self.model_dir / run_name / "weights"
        removed = []
        for artifact_name in ("best.pt", "last.pt"):
            artifact_path = weights_dir / artifact_name
            if artifact_path.exists():
                artifact_path.unlink()
                removed.append(str(artifact_path))

        current = weights_dir
        run_dir = self.model_dir / run_name
        for candidate in (current, run_dir):
            try:
                candidate.rmdir()
            except OSError:
                pass

        if removed:
            logger.info("Removed run-local training artifacts: %s", ", ".join(removed))

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
        from .utils import extract_frame, load_detections_jsonl, probe_video

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
