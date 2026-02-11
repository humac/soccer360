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
        self.require_tactical = exp_cfg.get("require_tactical", True)

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

        # Validate required outputs before writing metadata/copying artifacts.
        missing_required = []
        if not broadcast_src.exists():
            missing_required.append("broadcast.mp4")
        if self.require_tactical and not tactical_src.exists():
            missing_required.append("tactical_wide.mp4")
        if missing_required:
            raise RuntimeError(f"Missing required outputs: {', '.join(missing_required)}")

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
            self.highlights_base.mkdir(parents=True, exist_ok=True)
            highlights_dst = self._unique_named_dir(self.highlights_base, game_name)
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
            archive_dst = self._unique_file_path(self.archive_base, raw_path.name)
            shutil.move(str(raw_path), str(archive_dst))
            logger.info("Archived raw: %s", archive_dst)
        elif self.delete_raw and raw_path.exists():
            raw_path.unlink()
            logger.info("Deleted raw: %s", raw_path)

        return output_dir

    def _unique_output_dir(self, game_name: str) -> Path:
        """Create a unique output directory. Never overwrite silently."""
        base = self._unique_named_dir(self.output_base, game_name)
        return base

    @staticmethod
    def _unique_named_dir(parent: Path, name: str) -> Path:
        """Create a unique directory name by appending _runN when needed."""
        base = parent / name
        if not base.exists():
            return base

        # If directory already exists, append a run counter
        counter = 1
        while True:
            candidate = parent / f"{name}_run{counter}"
            if not candidate.exists():
                logger.info(
                    "Output dir %s exists, using %s", base, candidate
                )
                return candidate
            counter += 1

    @staticmethod
    def _split_name_suffixes(filename: str) -> tuple[str, str]:
        p = Path(filename)
        suffix = "".join(p.suffixes)
        stem = p.name[: -len(suffix)] if suffix else p.name
        return stem, suffix

    def _unique_file_path(self, parent: Path, filename: str) -> Path:
        """Create a unique file path by appending _runN when needed."""
        candidate = parent / filename
        if not candidate.exists():
            return candidate

        stem, suffix = self._split_name_suffixes(filename)
        counter = 1
        while True:
            candidate = parent / f"{stem}_run{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1
