"""Watch folder daemon: monitors /tank/ingest and triggers processing.

Uses watchdog (inotify on Linux) to detect new video files.
Processes one job at a time to avoid GPU contention.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .pipeline import Pipeline

logger = logging.getLogger("soccer360.watcher")

VIDEO_EXTENSIONS = {".mp4", ".insv", ".mov"}
STAGING_SUFFIXES = {".uploading", ".tmp", ".part"}


class VideoFileHandler(FileSystemEventHandler):
    """Watchdog handler that queues new video files for processing."""

    def __init__(
        self,
        job_queue: queue.Queue,
        scratch_dir: Path,
        stability_checks: int = 5,
        stability_interval: float = 10.0,
        ignore_suffixes: set[str] | None = None,
    ):
        super().__init__()
        self.job_queue = job_queue
        self.scratch_dir = scratch_dir
        self.stability_checks = stability_checks
        self.stability_interval = stability_interval
        self.ignore_suffixes = ignore_suffixes or STAGING_SUFFIXES
        self._processing = set()

    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Ignore staging/temp files (e.g. .uploading, .tmp, .part)
        if path.suffix.lower() in self.ignore_suffixes:
            return

        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            return

        if str(path) in self._processing:
            return
        self._processing.add(str(path))

        # Handle in a thread so we don't block the observer
        threading.Thread(
            target=self._handle_new_file,
            args=(path,),
            daemon=True,
        ).start()

    def _handle_new_file(self, path: Path):
        """Wait for file to finish writing, then copy to scratch and queue."""
        try:
            if not self._wait_stable(path):
                logger.warning("File did not stabilize: %s", path)
                return

            # Copy to scratch
            job_dir = self._copy_to_scratch(path)
            logger.info("Queued job: %s -> %s", path.name, job_dir)
            self.job_queue.put(str(job_dir))

        except Exception:
            logger.exception("Error handling new file: %s", path)
        finally:
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
        job_dir = self.scratch_dir / f"{timestamp}_{path.stem}"
        job_dir.mkdir(parents=True, exist_ok=True)

        dest = job_dir / path.name
        logger.info("Copying %s to %s (%.1f MB)", path.name, dest, path.stat().st_size / 1e6)
        shutil.copy2(str(path), str(dest))

        return dest


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

        self.job_queue: queue.Queue = queue.Queue()

    def run(self):
        """Start the watcher daemon. Blocks forever."""
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

        # Process any existing files in ingest
        self._process_existing()

        # Start watchdog observer
        handler = VideoFileHandler(
            self.job_queue,
            self.scratch_dir,
            stability_checks=self.stability_checks,
            stability_interval=self.stability_interval,
            ignore_suffixes=self.ignore_suffixes,
        )
        observer = Observer()
        observer.schedule(handler, str(self.ingest_dir), recursive=False)
        observer.start()

        logger.info("Watcher started. Waiting for files...")

        try:
            observer.join()
        except KeyboardInterrupt:
            logger.info("Shutting down watcher...")
            observer.stop()
            observer.join()

    def _process_existing(self):
        """Check for any video files already in ingest folder."""
        for path in self.ingest_dir.iterdir():
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                logger.info("Found existing file: %s", path.name)
                # Copy to scratch
                handler = VideoFileHandler(self.job_queue, self.scratch_dir)
                scratch_path = handler._copy_to_scratch(path)
                self.job_queue.put(str(scratch_path))

    def _process_loop(self):
        """Worker loop: process jobs sequentially from the queue."""
        while True:
            job_path = self.job_queue.get()
            logger.info("Processing job: %s", job_path)

            try:
                pipe = Pipeline(self.config)
                pipe.run(job_path, cleanup=True)
            except Exception:
                logger.exception("Pipeline failed for %s", job_path)
            finally:
                self.job_queue.task_done()
