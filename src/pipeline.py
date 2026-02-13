"""Pipeline orchestrator: coordinates all processing phases.

Data flow:
  input.mp4 -> detection (GPU) -> tracking (CPU) -> camera path (CPU)
  -> broadcast reframing (CPU parallel) -> tactical reframing (CPU parallel)
  -> highlights -> export to /tank/processed -> cleanup scratch
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from .active_learning import ActiveLearningExporter
from .camera import CameraPathGenerator
from .detector import Detector, resolve_model_path, resolve_model_path_v1
from .exporter import Exporter
from .hard_frames import HardFrameExporter
from .highlights import HighlightDetector
from .reframer import Reframer
from .tracker import BallStabilizer, Tracker
from .utils import VideoMeta, probe_video

logger = logging.getLogger("soccer360.pipeline")


class Pipeline:
    """End-to-end processing pipeline for 360 soccer video."""

    def __init__(self, config: dict):
        self.config = config
        self.scratch_base = Path(config["paths"]["scratch"])
        self._v1_mode = "detection" in config

        # Resolve model and determine operating mode
        if self._v1_mode:
            models_dir = config.get("paths", {}).get("models", "/app/models")
            resolved_path, self.mode = resolve_model_path_v1(
                config, models_dir=models_dir
            )
        else:
            resolved_path, self.mode = resolve_model_path(config)

        if self.mode == "normal":
            if self._v1_mode:
                config.setdefault("detection", {})["path"] = resolved_path
            else:
                config.setdefault("model", {})["path"] = resolved_path

            self.detector = Detector(config)

            if self._v1_mode:
                self.stabilizer = BallStabilizer(config)
                self.active_learner = ActiveLearningExporter(config)
                self.tracker = None
                self.hard_frame_exporter = None
            else:
                self.tracker = Tracker(config)
                self.hard_frame_exporter = HardFrameExporter(config)
                self.stabilizer = None
                self.active_learner = None
        else:
            logger.warning("NO_DETECT mode: no ball detection model found")
            self.detector = None
            self.tracker = None
            self.hard_frame_exporter = None
            self.stabilizer = None
            self.active_learner = None

        self.camera = CameraPathGenerator(config)
        self.reframer = Reframer(config)
        self.highlights = HighlightDetector(config)
        self.exporter = Exporter(config)

    def run(
        self,
        input_path: str | Path,
        cleanup: bool = True,
        ingest_source: str | Path | None = None,
    ):
        """Run the full processing pipeline on a single video.

        Args:
            input_path: Path to the 360 video file.
            cleanup: Remove scratch working directory after success.
            ingest_source: Original ingest file path (for post-success archival).
        """
        input_path = Path(input_path)
        start_time = datetime.now()

        # Create working directory on scratch
        job_id = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{input_path.stem}"
        work_dir = self.scratch_base / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("PIPELINE START: %s", input_path.name)
        logger.info("Job ID: %s", job_id)
        logger.info("Working dir: %s", work_dir)
        logger.info("=" * 60)

        try:
            # Probe video metadata
            meta = probe_video(input_path)
            logger.info(
                "Video: %dx%d, %.1f fps, %.1fs (%d frames), codec=%s",
                meta.width, meta.height, meta.fps, meta.duration,
                meta.total_frames, meta.codec,
            )
            logger.info("Operating mode: %s", self.mode)

            camera_path_file = work_dir / "camera_path.json"

            if self.mode == "normal":
                # Phase 1: Ball detection (GPU)
                logger.info("--- Phase 1: Ball Detection (GPU) ---")
                detections_path = work_dir / "detections.jsonl"
                processed_frames = self.detector.run_streaming(
                    str(input_path), meta, detections_path
                )

                tracks_path = work_dir / "tracks.json"

                if self._v1_mode:
                    # Phase 2: Ball stabilization (V1)
                    logger.info("--- Phase 2: Ball Stabilization (V1) ---")
                    total_frames = (
                        processed_frames
                        if processed_frames and processed_frames > 0
                        else meta.total_frames
                    )
                    tracking_events = self.stabilizer.run(
                        detections_path, tracks_path, meta.fps, total_frames=total_frames
                    )

                    # Phase 2.5: Active learning export (V1)
                    logger.info("--- Phase 2.5: Active Learning Export (V1) ---")
                    self.active_learner.run(
                        str(input_path), meta, detections_path, tracks_path,
                        work_dir, tracking_events=tracking_events, mode=self.mode,
                    )
                else:
                    # Phase 2: Ball tracking (legacy ByteTrack)
                    logger.info("--- Phase 2: Ball Tracking ---")
                    self.tracker.run(detections_path, tracks_path)

                    # Phase 2.5: Hard frame export (legacy)
                    logger.info("--- Phase 2.5: Hard Frame Export ---")
                    self.hard_frame_exporter.run(
                        str(input_path), meta, detections_path, tracks_path, work_dir
                    )

                # Phase 3: Camera path generation (CPU)
                logger.info("--- Phase 3: Camera Path Generation ---")
                self.camera.generate(tracks_path, meta, camera_path_file)
            else:
                # NO_DETECT mode: skip detection/tracking, static camera
                logger.info("--- NO_DETECT: Skipping phases 1-2, static camera path ---")
                self.camera.generate_static(meta, camera_path_file)
                tracks_path = None

            # Phase 4: Broadcast reframing (CPU, parallel)
            logger.info("--- Phase 4: Broadcast Reframing ---")
            broadcast_path = work_dir / "broadcast.mp4"
            self.reframer.render_broadcast(
                str(input_path), meta, camera_path_file, broadcast_path
            )

            # Phase 5: Tactical wide view (CPU, parallel)
            logger.info("--- Phase 5: Tactical Wide View ---")
            tactical_path = work_dir / "tactical_wide.mp4"
            self.reframer.render_tactical(str(input_path), meta, tactical_path)

            # Phase 6: Highlight detection and export
            logger.info("--- Phase 6: Highlights ---")
            highlights_dir = work_dir / "highlights"
            if self.mode == "normal" and tracks_path is not None:
                self.highlights.detect_and_export(
                    broadcast_path, meta, camera_path_file, tracks_path, highlights_dir
                )
            else:
                logger.info("Skipping highlights in NO_DETECT mode (no tracks)")

            # Phase 7: Export to final destination
            logger.info("--- Phase 7: Export ---")
            output_dir = self.exporter.finalize(
                work_dir, str(input_path), meta,
                processing_start=start_time,
                mode=self.mode,
                ingest_source=str(ingest_source) if ingest_source else None,
                job_id=job_id,
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info("=" * 60)
            logger.info(
                "PIPELINE COMPLETE: %s (%.1f min)", input_path.name, elapsed / 60
            )
            logger.info("Outputs: %s", output_dir)
            logger.info("=" * 60)

        except Exception:
            logger.exception("Pipeline failed for %s", input_path)
            raise

        finally:
            # Phase 8: Cleanup scratch
            if cleanup and work_dir.exists():
                logger.info("Cleaning up scratch: %s", work_dir)
                shutil.rmtree(work_dir, ignore_errors=True)
