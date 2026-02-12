"""Watch folder daemon: monitors /tank/ingest and triggers processing.

Uses watchdog (inotify on Linux) to detect new video files.
Processes one job at a time to avoid GPU contention.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .pipeline import Pipeline

logger = logging.getLogger("soccer360.watcher")

VIDEO_EXTENSIONS = {".mp4", ".insv", ".mov"}
STAGING_SUFFIXES = {".uploading", ".tmp", ".part"}


def _path_key(path: Path) -> str:
    return str(path.resolve(strict=False))


def _file_fingerprint(path: Path) -> dict[str, int] | None:
    """Stable-ish identity for dedupe across restarts and same-name replacements."""
    try:
        st = path.stat()
    except OSError:
        return None

    return {
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
        "ino": int(st.st_ino),
        "dev": int(st.st_dev),
    }


class ProcessedIngestStore:
    """Persistent map of ingest file path -> fingerprint for successful jobs."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._load()

    def is_processed(
        self, path: Path, fingerprint: dict[str, int] | None = None
    ) -> bool:
        key = _path_key(path)
        fp = fingerprint or _file_fingerprint(path)
        if fp is None:
            return False

        with self._lock:
            record = self._entries.get(key)
            return bool(record and record.get("fingerprint") == fp)

    def mark_processed(
        self,
        path: Path,
        fingerprint: dict[str, int],
        *,
        job_path: str | None = None,
    ):
        key = _path_key(path)
        entry = {
            "fingerprint": fingerprint,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "job_path": job_path,
        }
        with self._lock:
            self._entries[key] = entry
            self._persist_locked()

    def _load(self):
        if not self.state_file.exists():
            return

        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            entries = payload.get("entries", {})
            if isinstance(entries, dict):
                self._entries = entries
        except Exception:
            logger.exception("Failed to load watcher processed-state file: %s", self.state_file)

    def _persist_locked(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "entries": self._entries,
        }
        tmp = self.state_file.parent / (
            f".{self.state_file.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )

        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp, self.state_file)
            dir_fd = os.open(str(self.state_file.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if tmp.exists():
                tmp.unlink()


class VideoFileHandler(FileSystemEventHandler):
    """Watchdog handler that queues new video files for processing."""

    def __init__(
        self,
        job_queue: queue.Queue,
        scratch_dir: Path,
        stability_checks: int = 5,
        stability_interval: float = 10.0,
        ignore_suffixes: set[str] | None = None,
        max_copy_workers: int = 4,
        processed_store: ProcessedIngestStore | None = None,
    ):
        super().__init__()
        self.job_queue = job_queue
        self.scratch_dir = scratch_dir
        self.stability_checks = stability_checks
        self.stability_interval = stability_interval
        self.ignore_suffixes = ignore_suffixes or STAGING_SUFFIXES
        self.processed_store = processed_store
        self._processing: set[str] = set()
        self._processing_lock = threading.Lock()
        self._copy_pool = ThreadPoolExecutor(max_workers=max_copy_workers)

    def on_created(self, event: FileSystemEvent):
        self._handle_candidate(Path(event.src_path), is_directory=event.is_directory)

    def on_moved(self, event: FileSystemEvent):
        self._handle_candidate(Path(event.dest_path), is_directory=event.is_directory)

    def _handle_candidate(self, path: Path, is_directory: bool = False):
        """Apply filters/dedup and submit candidate for stabilization/copy."""
        if is_directory:
            return

        # Ignore hidden files (dotfiles like .DS_Store, .nfs*, etc.)
        if path.name.startswith("."):
            return

        # Ignore staging/temp files (e.g. .uploading, .tmp, .part)
        if path.suffix.lower() in self.ignore_suffixes:
            return

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            return

        if self.processed_store and self.processed_store.is_processed(path):
            logger.info("Skipping already processed ingest file: %s", path)
            return

        path_key = str(path)
        with self._processing_lock:
            if path_key in self._processing:
                return
            self._processing.add(path_key)

        # Use bounded pool to avoid unbounded daemon thread growth under burst ingest.
        try:
            self._copy_pool.submit(self._handle_new_file, path)
        except Exception:
            with self._processing_lock:
                self._processing.discard(path_key)
            raise

    def close(self):
        self._copy_pool.shutdown(wait=True)

    def _handle_new_file(self, path: Path):
        """Wait for file to finish writing, then copy to scratch and queue."""
        try:
            if not self._wait_stable(path):
                logger.warning("File did not stabilize: %s", path)
                return

            fingerprint = _file_fingerprint(path)
            if fingerprint is None:
                logger.warning("File missing before scratch copy, skipping: %s", path)
                return

            if self.processed_store and self.processed_store.is_processed(path, fingerprint):
                logger.info("Skipping already processed ingest file (post-stable): %s", path)
                return

            # Copy to scratch
            job_dir = self._copy_to_scratch(path)
            logger.info("Queued job: %s -> %s", path.name, job_dir)
            self.job_queue.put((str(job_dir), str(path), fingerprint))

        except Exception:
            logger.exception("Error handling new file: %s", path)
        finally:
            with self._processing_lock:
                self._processing.discard(str(path))

    def _wait_stable(self, path: Path) -> bool:
        """Wait for file size to stabilize (writer finished)."""
        prev_size = -1
        stable_count = 0

        for _ in range(self.stability_checks * 10):  # Max wait = 10x checks
            if not path.exists():
                return False

            size = path.stat().st_size
            if size == prev_size and size > 0:
                stable_count += 1
                if stable_count >= self.stability_checks:
                    return True
            else:
                stable_count = 0

            prev_size = size
            time.sleep(self.stability_interval)

        return False

    def _copy_to_scratch(self, path: Path) -> Path:
        """Copy video file to scratch working directory."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        job_dir = self.scratch_dir / f"{timestamp}_{path.stem}_{time.time_ns()}"
        job_dir.mkdir(parents=True, exist_ok=False)

        dest = job_dir / path.name
        logger.info("Copying %s to %s (%.1f MB)", path.name, dest, path.stat().st_size / 1e6)
        shutil.copy2(str(path), str(dest))

        return dest

    # Backward-compatible alias for tests/introspection.
    @property
    def _executor(self) -> ThreadPoolExecutor:
        return self._copy_pool


class WatcherDaemon:
    """Main daemon: watches ingest folder, processes jobs sequentially."""

    def __init__(self, config: dict):
        self.config = config
        self.ingest_dir = Path(config["paths"]["ingest"])
        self.scratch_dir = Path(config["paths"]["scratch"])

        watcher_cfg = config.get("watcher", {})
        self.stability_checks = watcher_cfg.get("stability_checks", 5)
        self.stability_interval = watcher_cfg.get("stability_interval_sec", 10.0)
        ignore = watcher_cfg.get("ignore_suffixes", [".uploading", ".tmp", ".part"])
        self.ignore_suffixes = set(ignore)
        self.max_copy_workers = watcher_cfg.get("max_copy_workers", 4)
        processed_state_cfg = watcher_cfg.get("processed_state_file")
        if processed_state_cfg:
            state_path = Path(processed_state_cfg)
            if not state_path.is_absolute():
                state_path = self.scratch_dir / state_path
        else:
            state_path = self.scratch_dir / "watcher_processed_ingest.json"
        self.processed_store = ProcessedIngestStore(state_path)

        self.job_queue: queue.Queue = queue.Queue()

    def _create_handler(self) -> VideoFileHandler:
        return VideoFileHandler(
            self.job_queue,
            self.scratch_dir,
            stability_checks=self.stability_checks,
            stability_interval=self.stability_interval,
            ignore_suffixes=self.ignore_suffixes,
            max_copy_workers=self.max_copy_workers,
            processed_store=self.processed_store,
        )

    def run(self):
        """Start the watcher daemon. Blocks forever."""
        Path("/scratch/work").mkdir(parents=True, exist_ok=True)
        self.ingest_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Soccer360 Watcher Daemon")
        logger.info("Watching: %s", self.ingest_dir)
        logger.info("Scratch:  %s", self.scratch_dir)
        logger.info("=" * 60)

        # Start processing worker thread
        worker = threading.Thread(target=self._process_loop, daemon=True)
        worker.start()

        # Process any existing files in ingest and start observer with same handler config.
        handler = self._create_handler()
        self._process_existing(handler)

        observer = Observer()
        observer.schedule(handler, str(self.ingest_dir), recursive=False)
        observer.start()

        logger.info("Watcher started. Waiting for files...")

        try:
            observer.join()
        except KeyboardInterrupt:
            logger.info("Shutting down watcher...")
            observer.stop()
        finally:
            observer.join()
            handler.close()

    def _process_existing(self, handler: VideoFileHandler):
        """Check for any video files already in ingest folder."""
        for path in self.ingest_dir.iterdir():
            if path.is_file():
                logger.info("Found existing candidate: %s", path.name)
                handler._handle_candidate(path)

    def _process_loop(self):
        """Worker loop: process jobs sequentially from the queue."""
        while True:
            job = self.job_queue.get()
            if isinstance(job, tuple) and len(job) == 3:
                job_path, ingest_source, ingest_fingerprint = job
            elif isinstance(job, tuple) and len(job) == 2:
                job_path, ingest_source = job
                ingest_fingerprint = None
            else:
                job_path = str(job)
                ingest_source = None
                ingest_fingerprint = None

            logger.info("Processing job: %s", job_path)
            succeeded = False

            try:
                pipe = Pipeline(self.config)
                pipe.run(job_path, cleanup=True, ingest_source=ingest_source)
                succeeded = True
            except Exception:
                logger.exception("Pipeline failed for %s", job_path)
            finally:
                if succeeded:
                    self._record_processed_ingest(
                        ingest_source,
                        ingest_fingerprint,
                        job_path=job_path,
                    )
                self.job_queue.task_done()

    def _record_processed_ingest(
        self,
        ingest_source: str | None,
        ingest_fingerprint: dict[str, int] | None,
        *,
        job_path: str | None = None,
    ):
        """Persist ingest completion marker to avoid restart reprocessing loops."""
        if not ingest_source:
            return

        source_path = Path(ingest_source)
        fingerprint = ingest_fingerprint or _file_fingerprint(source_path)
        if fingerprint is None:
            logger.warning(
                "Unable to fingerprint completed ingest source for dedupe: %s",
                ingest_source,
            )
            return

        try:
            self.processed_store.mark_processed(
                source_path,
                fingerprint,
                job_path=job_path,
            )
        except Exception:
            logger.exception(
                "Failed to persist watcher dedupe marker for completed ingest: %s",
                ingest_source,
            )
