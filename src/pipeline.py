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
from .metrics import PhaseTimer, gpu_utilization_snapshot
from .player_cluster import PlayerClusterComputer
from .reframer import Reframer

from contextlib import contextmanager
from .tracker import BallStabilizer, Tracker
from .utils import VideoMeta, probe_video

logger = logging.getLogger("soccer360.pipeline")


class Pipeline:
    """End-to-end processing pipeline for 360 soccer video."""

    def __init__(self, config: dict, event_bus=None):
        self.config = config
        self.scratch_base = Path(config["paths"]["scratch"])
        self._v1_mode = "detection" in config
        self.event_bus = event_bus

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

        # Center of play: player cluster tracking for hybrid camera
        cop_cfg = config.get("center_of_play", {})
        if cop_cfg.get("enabled", False) and self.mode == "normal":
            self.player_cluster = PlayerClusterComputer(config)
        else:
            self.player_cluster = None

    @contextmanager
    def _tracked_phase(self, timer: PhaseTimer, job_id: str, phase_name: str):
        """Wrap a phase with both timing and event bus emission."""
        if self.event_bus:
            self.event_bus.phase_started(job_id, phase_name)
        try:
            with timer.phase(phase_name):
                yield
            if self.event_bus:
                elapsed = timer._timings.get(phase_name)
                self.event_bus.phase_completed(job_id, phase_name, duration_sec=elapsed)
        except Exception:
            if self.event_bus:
                self.event_bus.phase_failed(job_id, phase_name)
            raise

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

        if self.event_bus:
            self.event_bus.job_created(job_id, str(input_path))

        timer = PhaseTimer()

        try:
            # Probe video metadata
            meta = probe_video(input_path)
            logger.info(
                "Video: %dx%d, %.1f fps, %.1fs (%d frames), codec=%s",
                meta.width, meta.height, meta.fps, meta.duration,
                meta.total_frames, meta.codec,
            )
            logger.info("Operating mode: %s", self.mode)

            if self.event_bus:
                self.event_bus.job_started(job_id, mode=self.mode)

            camera_path_file = work_dir / "camera_path.json"

            if self.mode == "normal":
                # Decision: confirm mode before starting detection
                if self.event_bus:
                    mode_label = "V1 bootstrap" if self._v1_mode else "legacy"
                    self.event_bus.request_decision(
                        job_id,
                        "mode_confirm",
                        f"Proceeding in {mode_label} mode. Input: {input_path.name} "
                        f"({meta.total_frames} frames, {meta.duration:.0f}s). Continue?",
                        options=["continue", "cancel"],
                        default_option="continue",
                        timeout_sec=30,
                    )

                # Phase 1: Ball detection (GPU)
                logger.info("--- Phase 1: Ball Detection (GPU) ---")
                detections_path = work_dir / "detections.jsonl"
                with self._tracked_phase(timer, job_id, "detection"):
                    processed_frames = self.detector.run_streaming(
                        str(input_path), meta, detections_path
                    )

                # Capture GPU snapshot right after detection (GPU-intensive phase)
                timer.record_stat("gpu_snapshot_post_detection", gpu_utilization_snapshot())

                # Count detections from JSONL
                detection_count = sum(1 for _ in open(detections_path))
                timer.record_stat("detection_count", detection_count)
                timer.record_stat("frames_processed", processed_frames or meta.total_frames)

                # Decision: review detection results before continuing
                if self.event_bus:
                    total = processed_frames or meta.total_frames
                    coverage = (detection_count / total * 100) if total > 0 else 0
                    self.event_bus.request_decision(
                        job_id,
                        "post_detection_review",
                        f"Detection complete: {detection_count} detections in {total} frames "
                        f"({coverage:.1f}% coverage). Continue to tracking?",
                        options=["continue", "cancel"],
                        default_option="continue",
                        timeout_sec=60,
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
                    with self._tracked_phase(timer, job_id, "tracking"):
                        tracking_events = self.stabilizer.run(
                            detections_path, tracks_path, meta.fps, total_frames=total_frames
                        )

                    # Phase 2.5: Active learning export (V1)
                    logger.info("--- Phase 2.5: Active Learning Export (V1) ---")
                    with self._tracked_phase(timer, job_id, "hard_frames"):
                        self.active_learner.run(
                            str(input_path), meta, detections_path, tracks_path,
                            work_dir, tracking_events=tracking_events, mode=self.mode,
                        )
                else:
                    # Phase 2: Ball tracking (legacy ByteTrack)
                    logger.info("--- Phase 2: Ball Tracking ---")
                    with self._tracked_phase(timer, job_id, "tracking"):
                        self.tracker.run(detections_path, tracks_path)

                    # Phase 2.5: Hard frame export (legacy)
                    logger.info("--- Phase 2.5: Hard Frame Export ---")
                    with self._tracked_phase(timer, job_id, "hard_frames"):
                        self.hard_frame_exporter.run(
                            str(input_path), meta, detections_path, tracks_path, work_dir
                        )

                # Decision: prompt to review hard frames in Label Studio
                if self.event_bus:
                    hard_frames_manifest = work_dir / "hard_frames.json"
                    hf_count = 0
                    if hard_frames_manifest.exists():
                        import json as _json_hf
                        try:
                            hf_data = _json_hf.loads(hard_frames_manifest.read_text())
                            hf_count = hf_data.get("exported_count", 0)
                        except Exception:
                            pass
                    if hf_count > 0:
                        self.event_bus.request_decision(
                            job_id,
                            "hard_frame_labeling",
                            f"{hf_count} hard frames exported for labeling. "
                            f"Open Label Studio (port 8080) to review and annotate them. "
                            f"Pipeline will continue rendering in the background.",
                            options=["continue", "pause"],
                            default_option="continue",
                            timeout_sec=120,
                        )

                # Record tracking quality stats
                import json as _json
                tracks_data = _json.loads(tracks_path.read_text())
                ball_found = sum(
                    1 for t in tracks_data
                    if t.get("ball") is not None or t.get("x") is not None
                )
                timer.record_stat("track_frames_total", len(tracks_data))
                timer.record_stat("track_frames_with_ball", ball_found)

                # Phase 2.7: Player cluster (center of play)
                player_cluster_path = None
                if self.player_cluster is not None:
                    logger.info("--- Phase 2.7: Player Cluster (Center of Play) ---")
                    player_cluster_path = work_dir / "player_cluster.json"
                    target_frames = (
                        processed_frames
                        if processed_frames and processed_frames > 0
                        else meta.total_frames
                    )
                    with self._tracked_phase(timer, job_id, "player_cluster"):
                        self.player_cluster.run(
                            detections_path, player_cluster_path, target_frames
                        )

                # Phase 3: Camera path generation (CPU)
                logger.info("--- Phase 3: Camera Path Generation ---")
                with self._tracked_phase(timer, job_id, "camera"):
                    self.camera.generate(
                        tracks_path, meta, camera_path_file,
                        player_cluster_path=player_cluster_path,
                    )
            else:
                # NO_DETECT mode: skip detection/tracking, static camera
                logger.info("--- NO_DETECT: Skipping phases 1-2, static camera path ---")
                with self._tracked_phase(timer, job_id, "camera"):
                    self.camera.generate_static(meta, camera_path_file)
                tracks_path = None

            # Phase 4: Broadcast reframing (CPU, parallel)
            logger.info("--- Phase 4: Broadcast Reframing ---")
            broadcast_path = work_dir / "broadcast.mp4"
            with self._tracked_phase(timer, job_id, "broadcast_reframe"):
                self.reframer.render_broadcast(
                    str(input_path), meta, camera_path_file, broadcast_path
                )

            # Phase 5: Tactical wide view (CPU, parallel)
            logger.info("--- Phase 5: Tactical Wide View ---")
            tactical_path = work_dir / "tactical_wide.mp4"
            with self._tracked_phase(timer, job_id, "tactical_reframe"):
                self.reframer.render_tactical(str(input_path), meta, tactical_path)

            # Phase 6: Highlight detection and export
            logger.info("--- Phase 6: Highlights ---")
            highlights_dir = work_dir / "highlights"
            with self._tracked_phase(timer, job_id, "highlights"):
                if self.mode == "normal" and tracks_path is not None:
                    self.highlights.detect_and_export(
                        broadcast_path, meta, camera_path_file, tracks_path, highlights_dir
                    )
                else:
                    logger.info("Skipping highlights in NO_DETECT mode (no tracks)")

            # Phase 7: Export to final destination
            logger.info("--- Phase 7: Export ---")
            with self._tracked_phase(timer, job_id, "export"):
                output_dir = self.exporter.finalize(
                    work_dir, str(input_path), meta,
                    processing_start=start_time,
                    mode=self.mode,
                    ingest_source=str(ingest_source) if ingest_source else None,
                    job_id=job_id,
                    phase_metrics=timer.to_dict(),
                )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info("=" * 60)
            logger.info(
                "PIPELINE COMPLETE: %s (%.1f min)", input_path.name, elapsed / 60
            )
            logger.info("Outputs: %s", output_dir)
            logger.info("=" * 60)

            if self.event_bus:
                self.event_bus.job_completed(job_id)

        except Exception as exc:
            logger.exception("Pipeline failed for %s", input_path)
            if self.event_bus:
                self.event_bus.job_failed(job_id, error=str(exc))
            raise

        finally:
            # Phase 8: Cleanup scratch
            if cleanup and work_dir.exists():
                logger.info("Cleaning up scratch: %s", work_dir)
                shutil.rmtree(work_dir, ignore_errors=True)
