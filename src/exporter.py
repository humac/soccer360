"""Output finalization: move results to /tank/processed, write metadata."""

from __future__ import annotations

import errno
import logging
import os
import shutil
import time
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

        # Ingest archival config
        ing = config.get("ingest", {})
        self.ingest_archive_on_success = ing.get("archive_on_success", False)
        self.ingest_archive_dir = Path(
            ing.get("archive_dir", paths.get("archive_raw", "/tank/archive_raw"))
        )
        self.ingest_archive_mode = ing.get("archive_mode", "leave")
        self.ingest_name_template = ing.get(
            "archive_name_template", "{match}_{job_id}{ext}"
        )
        self.ingest_collision = ing.get("archive_collision", "suffix")

    def finalize(
        self,
        work_dir: Path,
        input_path: str,
        meta: VideoMeta,
        processing_start: datetime | None = None,
        mode: str = "normal",
        ingest_source: str | None = None,
        job_id: str | None = None,
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

        # Preserve intermediate artifacts (conditional on mode)
        if mode == "normal":
            artifacts = [
                "detections.jsonl",
                "tracks.json",
                "camera_path.json",
                "foi_meta.json",
                "hard_frames.json",
            ]
        else:
            artifacts = ["camera_path.json"]
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
            "job_id": job_id,
            "game_name": game_name,
            "duration_sec": meta.duration,
            "fps": meta.fps,
            "resolution": f"{meta.width}x{meta.height}",
            "processed_at": now.isoformat(),
            "processing_duration_sec": (
                (now - processing_start).total_seconds()
                if processing_start else None
            ),
            "mode": mode,
            "outputs": outputs,
            "artifacts": artifacts,
        }

        # Ingest archival (post-success only)
        archive_result = self._archive_ingest(ingest_source, game_name, job_id)
        summary["ingest_archive"] = archive_result
        summary["ingest_source_path"] = archive_result.get("source_path")
        summary["ingest_archived_path"] = archive_result.get("archived_path")
        summary["ingest_archive_destination_path"] = archive_result.get("destination_path")
        summary["ingest_archive_mode"] = archive_result.get("mode")
        summary["ingest_archive_collision"] = archive_result.get("collision_policy")
        summary["ingest_archive_status"] = archive_result["status"]
        summary["ingest_archive_error"] = archive_result.get("error")

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

    # ------------------------------------------------------------------
    # Ingest archival
    # ------------------------------------------------------------------

    def _archive_ingest(
        self,
        ingest_source: str | None,
        game_name: str,
        job_id: str | None,
    ) -> dict:
        """Archive the original ingest file after a successful run.

        Returns a dict with at least ``status`` (success|skipped|failed)
        and optionally ``archived_path``.  Never raises -- archival
        failure must not mark the pipeline as failed.
        """
        result = {
            "status": "skipped",
            "reason": None,
            "source_path": str(ingest_source) if ingest_source else None,
            "destination_path": None,
            "archived_path": None,
            "mode": self.ingest_archive_mode,
            "collision_policy": self.ingest_collision,
            "archive_dir": str(self.ingest_archive_dir),
            "job_id": job_id,
            "error": None,
        }
        if (
            not self.ingest_archive_on_success
            or self.ingest_archive_mode == "leave"
            or ingest_source is None
        ):
            result["reason"] = "disabled_or_no_source"
            return result

        src = Path(ingest_source)
        if not src.exists():
            logger.warning("Ingest source missing, cannot archive: %s", src)
            result["reason"] = "source_missing"
            return result

        try:
            self.ingest_archive_dir.mkdir(parents=True, exist_ok=True)

            base_dest = self._build_archive_destination(src, game_name, job_id)
            result["destination_path"] = str(base_dest)

            # Retry suffix resolution if a concurrent writer grabs the same destination.
            for _ in range(1000):
                dest = self._resolve_ingest_collision(base_dest)
                if dest is None:
                    result["reason"] = "collision_skip"
                    return result

                result["destination_path"] = str(dest)
                dest.parent.mkdir(parents=True, exist_ok=True)
                overwrite = self.ingest_collision == "overwrite"

                try:
                    if self.ingest_archive_mode == "move":
                        self._safe_move(src, dest, overwrite=overwrite)
                    elif self.ingest_archive_mode == "copy":
                        self._safe_copy(src, dest, overwrite=overwrite)
                    else:
                        logger.warning(
                            "Unknown archive_mode '%s', skipping", self.ingest_archive_mode
                        )
                        result["reason"] = "unknown_mode"
                        return result
                    break
                except FileExistsError:
                    if self.ingest_collision == "suffix":
                        continue
                    if self.ingest_collision == "skip":
                        result["reason"] = "collision_skip"
                        return result
                    raise
            else:
                raise RuntimeError("Unable to resolve unique archive filename after 1000 attempts")

            logger.info(
                "Archived ingest file (job_id=%s): %s -> %s (mode=%s, collision=%s)",
                job_id, src, dest, self.ingest_archive_mode, self.ingest_collision,
            )
            result["status"] = "success"
            result["reason"] = "archived"
            result["archived_path"] = str(dest)
            return result

        except Exception as exc:
            result["status"] = "failed"
            result["reason"] = "exception"
            result["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "Failed to archive ingest file (job_id=%s, src=%s, dst=%s)",
                job_id,
                src,
                result.get("destination_path"),
            )
            return result

    def _build_archive_destination(
        self,
        src: Path,
        game_name: str,
        job_id: str | None,
    ) -> Path:
        """Render and validate archive destination stays inside archive_dir."""
        _, suffix = self._split_name_suffixes(src.name)
        try:
            rendered = self.ingest_name_template.format(
                match=game_name,
                job_id=job_id or "unknown",
                ext=suffix,
            ).strip()
        except Exception as exc:
            raise ValueError(
                f"Invalid archive_name_template '{self.ingest_name_template}'"
            ) from exc

        if not rendered:
            raise ValueError("archive_name_template rendered an empty name")

        rendered_path = Path(rendered)
        if rendered_path.is_absolute():
            raise ValueError("archive_name_template must not render an absolute path")

        archive_root = self.ingest_archive_dir.resolve()
        destination = (archive_root / rendered_path).resolve()
        if not destination.is_relative_to(archive_root):
            raise ValueError("archive_name_template escapes archive_dir")

        return destination

    def _resolve_ingest_collision(self, dest: Path) -> Path | None:
        """Apply collision policy.  Returns adjusted path or None for skip."""
        if not dest.exists():
            return dest

        if self.ingest_collision == "skip":
            logger.warning(
                "Archive destination exists, skipping (collision=skip): %s", dest
            )
            return None

        if self.ingest_collision == "overwrite":
            logger.warning(
                "Archive destination exists, overwriting (collision=overwrite): %s", dest
            )
            return dest

        # Default: suffix (_01, _02, ...)
        stem, suffix = self._split_name_suffixes(dest.name)
        counter = 1
        while True:
            candidate = dest.parent / f"{stem}_{counter:02d}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    @classmethod
    def _safe_move(cls, src: Path, dst: Path, *, overwrite: bool = False):
        """Move file with cross-filesystem fallback.

        On same filesystem, uses ``os.rename``/``os.replace``. On cross-filesystem
        moves, performs copy-to-temp + fsync + atomic destination publish + source delete.
        """
        try:
            if overwrite:
                os.replace(str(src), str(dst))
            else:
                # Atomic no-overwrite move when hard-links are supported.
                os.link(str(src), str(dst))
                src.unlink()
            return
        except FileExistsError:
            raise
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
                raise

        # Cross-filesystem move: copy safely, then delete source.
        cls._safe_copy(src, dst, overwrite=overwrite)
        src.unlink()

    @classmethod
    def _safe_copy(cls, src: Path, dst: Path, *, overwrite: bool = False):
        """Copy file to destination with fsync and atomic publish."""
        src_size = src.stat().st_size
        tmp = dst.parent / f".{dst.name}.tmp-{os.getpid()}-{time.time_ns()}"

        try:
            shutil.copy2(str(src), str(tmp))
            cls._fsync_path(tmp)

            if tmp.stat().st_size != src_size:
                raise RuntimeError(f"Size mismatch after copy: {src} -> {tmp}")

            if overwrite:
                os.replace(str(tmp), str(dst))
            else:
                # Atomic no-overwrite publish. Raises FileExistsError on race.
                try:
                    os.link(str(tmp), str(dst))
                    tmp.unlink()
                except FileExistsError:
                    raise
                except OSError as exc:
                    if exc.errno not in {errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP}:
                        raise
                    if dst.exists():
                        raise FileExistsError(str(dst))
                    # Fallback for filesystems that do not support hard-links.
                    os.replace(str(tmp), str(dst))

            cls._fsync_path(dst)
            cls._fsync_dir(dst.parent)

            if dst.stat().st_size != src_size:
                raise RuntimeError(f"Size mismatch after publish: {src} -> {dst}")
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _fsync_path(path: Path):
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _fsync_dir(path: Path):
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
