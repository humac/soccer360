"""Output finalization: move results to /tank/processed, write metadata."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from .utils import VideoMeta, write_json

logger = logging.getLogger("soccer360.exporter")


class Exporter:
    """Organize final outputs and write processing metadata.

    Preserves all intermediate artifacts (detections, tracks, camera_path,
    ffprobe metadata, config snapshot) alongside final outputs.
    Never overwrites existing outputs silently -- uses timestamped dirs.
    """

    def __init__(self, config: dict):
        self.config = config
        paths = config.get("paths", {})
        exp_cfg = config.get("exporter", {})

        self.output_base = Path(paths.get("processed", "/tank/processed"))
        self.highlights_base = Path(paths.get("highlights", "/tank/highlights"))
        self.archive_base = Path(paths.get("archive_raw", "/tank/archive_raw"))

        self.archive_raw = exp_cfg.get("archive_raw", False)
        self.delete_raw = exp_cfg.get("delete_raw", False)

    def finalize(
        self,
        work_dir: Path,
        input_path: str,
        meta: VideoMeta,
        processing_start: datetime | None = None,
    ):
        """Move outputs from scratch to final destinations."""
        game_name = Path(input_path).stem
        output_dir = self._unique_output_dir(game_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Move main outputs
        broadcast_src = work_dir / "broadcast.mp4"
        tactical_src = work_dir / "tactical_wide.mp4"

        outputs = {}

        if broadcast_src.exists():
            dst = output_dir / "broadcast.mp4"
            shutil.move(str(broadcast_src), str(dst))
            outputs["broadcast"] = str(dst)
            logger.info("Exported broadcast: %s", dst)

        if tactical_src.exists():
            dst = output_dir / "tactical_wide.mp4"
            shutil.move(str(tactical_src), str(dst))
            outputs["tactical"] = str(dst)
            logger.info("Exported tactical: %s", dst)

        # Move highlights
        highlights_src = work_dir / "highlights"
        highlights_dst = None
        if highlights_src.exists() and any(highlights_src.iterdir()):
            highlights_dst = self.highlights_base / game_name
            if highlights_dst.exists():
                shutil.rmtree(highlights_dst)
            shutil.copytree(str(highlights_src), str(highlights_dst))
            outputs["highlights"] = str(highlights_dst)
            logger.info("Exported highlights: %s", highlights_dst)

        # Preserve ALL intermediate artifacts
        artifacts = [
            "detections.jsonl",
            "tracks.json",
            "camera_path.json",
            "foi_meta.json",
        ]
        for name in artifacts:
            src = work_dir / name
            if src.exists():
                shutil.copy2(str(src), str(output_dir / name))
                logger.info("Preserved artifact: %s", name)

        # Save ffprobe metadata
        probe_data = {
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "duration": meta.duration,
            "total_frames": meta.total_frames,
            "codec": meta.codec,
        }
        write_json(probe_data, output_dir / "ffprobe_meta.json")

        # Save config snapshot used for this run
        write_json(self.config, output_dir / "config_snapshot.json")

        # Write metadata summary
        now = datetime.now()
        summary = {
            "source": str(input_path),
            "game_name": game_name,
            "duration_sec": meta.duration,
            "fps": meta.fps,
            "resolution": f"{meta.width}x{meta.height}",
            "processed_at": now.isoformat(),
            "processing_duration_sec": (
                (now - processing_start).total_seconds()
                if processing_start else None
            ),
            "outputs": outputs,
            "artifacts": artifacts,
        }
        write_json(summary, output_dir / "metadata.json")
        logger.info("Metadata written to %s", output_dir / "metadata.json")

        # Handle raw file
        raw_path = Path(input_path)
        if self.archive_raw and raw_path.exists():
            self.archive_base.mkdir(parents=True, exist_ok=True)
            archive_dst = self.archive_base / raw_path.name
            shutil.move(str(raw_path), str(archive_dst))
            logger.info("Archived raw: %s", archive_dst)
        elif self.delete_raw and raw_path.exists():
            raw_path.unlink()
            logger.info("Deleted raw: %s", raw_path)

        return output_dir

    def _unique_output_dir(self, game_name: str) -> Path:
        """Create a unique output directory. Never overwrite silently."""
        base = self.output_base / game_name
        if not base.exists():
            return base

        # If directory already exists, append a run counter
        counter = 1
        while True:
            candidate = self.output_base / f"{game_name}_run{counter}"
            if not candidate.exists():
                logger.info(
                    "Output dir %s exists, using %s", base, candidate
                )
                return candidate
            counter += 1
