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
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .pipeline import Pipeline

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

logger = logging.getLogger("soccer360.watcher")

VIDEO_EXTENSIONS = {".mp4", ".insv", ".mov"}
STAGING_SUFFIXES = {".uploading", ".tmp", ".part"}


def _path_key(path: Path) -> str:
    return str(path.resolve(strict=False))


def _normalize_fingerprint(
    fingerprint: Mapping[str, object] | None,
) -> dict[str, int]:
    """Normalize fingerprint fields for stable comparisons/storage."""
    if not isinstance(fingerprint, Mapping):
        return {}

    normalized: dict[str, int] = {}
    for key in ("size", "mtime_ns"):
        value = fingerprint.get(key)
        if not isinstance(value, int) or value < 0:
            return {}
        normalized[key] = int(value)

    ctime_ns = fingerprint.get("ctime_ns")
    if isinstance(ctime_ns, int) and ctime_ns > 0:
        normalized["ctime_ns"] = int(ctime_ns)

    # Some network filesystems report unstable/placeholder inode/device IDs.
    ino = fingerprint.get("ino")
    if isinstance(ino, int) and ino > 0:
        normalized["ino"] = int(ino)

    dev = fingerprint.get("dev")
    if isinstance(dev, int) and dev > 0:
        normalized["dev"] = int(dev)

    return normalized


def compute_fingerprint(path: Path) -> dict[str, int] | None:
    """Compute a JSON-serializable ingest fingerprint with defensive fields."""
    try:
        st = path.stat()
    except OSError:
        return None

    fingerprint: dict[str, int] = {
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }

    ctime_ns = int(getattr(st, "st_ctime_ns", 0))
    if ctime_ns > 0:
        fingerprint["ctime_ns"] = ctime_ns

    ino = int(getattr(st, "st_ino", 0))
    if ino > 0:
        fingerprint["ino"] = ino

    dev = int(getattr(st, "st_dev", 0))
    if dev > 0:
        fingerprint["dev"] = dev

    return _normalize_fingerprint(fingerprint)


def fingerprint_key(fingerprint: Mapping[str, object] | None) -> str:
    """Stable key for material fingerprint comparison.

    ctime is intentionally excluded from the key because it can be noisy on
    some filesystems and metadata operations.
    """
    normalized = _normalize_fingerprint(fingerprint)
    if not normalized:
        return ""

    material: dict[str, int] = {
        "size": normalized["size"],
        "mtime_ns": normalized["mtime_ns"],
    }
    if "ino" in normalized:
        material["ino"] = normalized["ino"]
    if "dev" in normalized:
        material["dev"] = normalized["dev"]
    return json.dumps(material, sort_keys=True, separators=(",", ":"))


def _file_fingerprint(path: Path) -> dict[str, int] | None:
    """Backward-compatible alias."""
    return compute_fingerprint(path)


class IngestStateStore:
    """Persistent map of ingest path -> latest successful fingerprint."""

    def __init__(self, state_file: Path, *, max_entries: int = 50000):
        self.state_file = state_file
        self.lock_file = state_file.with_name(f".{state_file.name}.lock")
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._load()

    def is_processed(
        self, path: Path, fingerprint: Mapping[str, object] | None = None
    ) -> bool:
        key = _path_key(path)
        fp = _normalize_fingerprint(fingerprint) if fingerprint else compute_fingerprint(path)
        if fp is None or not fp:
            return False

        fp_key = fingerprint_key(fp)
        if not fp_key:
            return False

        with self._lock:
            record = self._entries.get(key)
            if not isinstance(record, dict):
                return False
            return fingerprint_key(record.get("fingerprint")) == fp_key

    def mark_processed(
        self,
        path: Path,
        fingerprint: Mapping[str, object],
        *,
        job_path: str | None = None,
    ):
        key = _path_key(path)
        fp = _normalize_fingerprint(fingerprint)
        if not fp:
            logger.warning(
                "Skipping watcher dedupe update with invalid fingerprint: %s",
                path,
            )
            return

        entry = {
            "fingerprint": fp,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "job_path": job_path,
        }

        with self._lock:
            try:
                with self._file_lock():
                    self._entries = self._read_entries_locked()
                    self._entries[key] = entry
                    self._prune_locked()
                    self._persist_locked()
            except Exception:
                # Dedupe bookkeeping must never stop processing.
                logger.warning(
                    "Failed to persist watcher processed-state update: source=%s job_path=%s",
                    path,
                    job_path,
                    exc_info=True,
                )

    def _load(self):
        with self._lock:
            try:
                with self._file_lock():
                    self._entries = self._read_entries_locked()
                    if self._prune_locked() > 0:
                        try:
                            self._persist_locked()
                        except Exception:
                            logger.warning(
                                "Failed to persist pruned watcher processed-state file: %s",
                                self.state_file,
                                exc_info=True,
                            )
            except Exception:
                # If loading fails, continue with empty state.
                self._entries = {}
                logger.warning(
                    "Failed to initialize watcher processed-state store: %s",
                    self.state_file,
                    exc_info=True,
                )

    @contextmanager
    def _file_lock(self):
        """Best-effort inter-process lock using a sidecar lock file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        fd = None
        try:
            fd = os.open(str(self.lock_file), os.O_CREAT | os.O_RDWR, 0o644)
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
            logger.warning(
                "Failed to acquire watcher processed-state file lock: %s",
                self.lock_file,
                exc_info=True,
            )
        try:
            yield
        finally:
            if fd is not None:
                try:
                    if fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def _read_entries_locked(self) -> dict[str, dict]:
        if not self.state_file.exists():
            return {}

        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("state payload must be an object")
            raw_entries = payload.get("entries", {})
            if not isinstance(raw_entries, dict):
                raise ValueError("state payload 'entries' must be an object")

            entries: dict[str, dict] = {}
            for key, value in raw_entries.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                fp = _normalize_fingerprint(value.get("fingerprint"))
                if not fp:
                    continue
                processed_at = value.get("processed_at")
                if not isinstance(processed_at, str):
                    processed_at = ""
                job_path = value.get("job_path")
                if job_path is not None and not isinstance(job_path, str):
                    job_path = str(job_path)
                entries[key] = {
                    "fingerprint": fp,
                    "processed_at": processed_at,
                    "job_path": job_path,
                }
            return entries
        except (json.JSONDecodeError, ValueError) as exc:
            self._quarantine_corrupt_locked(exc)
            return {}
        except FileNotFoundError:
            return {}
        except OSError:
            logger.warning(
                "Failed to read watcher processed-state file: %s",
                self.state_file,
                exc_info=True,
            )
            return {}
        except Exception as exc:
            self._quarantine_corrupt_locked(exc)
            return {}

    def _quarantine_corrupt_locked(self, error: Exception):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        corrupt = self.state_file.with_name(f"{self.state_file.name}.corrupt.{ts}")
        counter = 1
        while corrupt.exists():
            corrupt = self.state_file.with_name(
                f"{self.state_file.name}.corrupt.{ts}.{counter}"
            )
            counter += 1

        try:
            os.replace(self.state_file, corrupt)
            logger.error(
                "Corrupt watcher processed-state file moved aside: %s -> %s (error=%s)",
                self.state_file,
                corrupt,
                error,
            )
        except Exception:
            logger.error(
                "Failed to quarantine corrupt watcher processed-state file: %s (error=%s)",
                self.state_file,
                error,
                exc_info=True,
            )

    def _prune_locked(self) -> int:
        if self.max_entries <= 0:
            return 0
        if len(self._entries) <= self.max_entries:
            return 0

        ordered = sorted(
            self._entries.items(),
            key=lambda item: (item[1].get("processed_at") or "", item[0]),
            reverse=True,
        )
        keep_keys = {key for key, _ in ordered[: self.max_entries]}
        removed = 0
        for key in list(self._entries.keys()):
            if key not in keep_keys:
                del self._entries[key]
                removed += 1
        return removed

    def _persist_locked(self):
        payload = {"version": 1, "entries": self._entries}
        tmp = self.state_file.parent / (
            f".{self.state_file.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )

        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp, self.state_file)
            self._fsync_dir_best_effort(self.state_file.parent)
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _fsync_dir_best_effort(path: Path):
        try:
            dir_fd = os.open(str(path), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            # Best effort on filesystems/platforms that do not support dir fsync.
            pass
        finally:
            os.close(dir_fd)


# Backward-compatible alias.
ProcessedIngestStore = IngestStateStore


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

            fingerprint = compute_fingerprint(path)
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
        self.output_dir = Path(config.get("paths", {}).get("processed", "/tank/processed"))

        watcher_cfg = config.get("watcher", {})
        self.stability_checks = watcher_cfg.get("stability_checks", 5)
        self.stability_interval = watcher_cfg.get("stability_interval_sec", 10.0)
        ignore = watcher_cfg.get("ignore_suffixes", [".uploading", ".tmp", ".part"])
        self.ignore_suffixes = set(ignore)
        self.max_copy_workers = watcher_cfg.get("max_copy_workers", 4)
        max_entries_cfg = watcher_cfg.get("processed_state_max_entries", 50000)
        try:
            self.processed_state_max_entries = int(max_entries_cfg)
        except (TypeError, ValueError):
            self.processed_state_max_entries = 50000
            logger.warning(
                "Invalid watcher.processed_state_max_entries=%r; using %d",
                max_entries_cfg,
                self.processed_state_max_entries,
            )

        processed_state_cfg = watcher_cfg.get("processed_state_file")
        state_base = self.output_dir / ".state"
        if processed_state_cfg:
            state_path = Path(processed_state_cfg)
            if not state_path.is_absolute():
                state_path = state_base / state_path
        else:
            state_path = state_base / "watcher_processed_ingest.json"
        self.processed_store = IngestStateStore(
            state_path,
            max_entries=self.processed_state_max_entries,
        )

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
            try:
                self._process_job(job)
            except Exception:
                logger.exception("Unhandled watcher worker error for job: %r", job)
            finally:
                self.job_queue.task_done()

    @staticmethod
    def _parse_job(
        job: object,
    ) -> tuple[str, str | None, dict[str, int] | None, int]:
        if isinstance(job, tuple) and len(job) == 4:
            job_path, ingest_source, ingest_fingerprint, requeue_count = job
            try:
                retries = int(requeue_count)
            except (TypeError, ValueError):
                retries = 0
            return (
                str(job_path),
                str(ingest_source) if ingest_source is not None else None,
                _normalize_fingerprint(ingest_fingerprint),
                retries,
            )
        if isinstance(job, tuple) and len(job) == 3:
            job_path, ingest_source, ingest_fingerprint = job
            return (
                str(job_path),
                str(ingest_source) if ingest_source is not None else None,
                _normalize_fingerprint(ingest_fingerprint),
                0,
            )
        if isinstance(job, tuple) and len(job) == 2:
            job_path, ingest_source = job
            return (
                str(job_path),
                str(ingest_source) if ingest_source is not None else None,
                None,
                0,
            )
        return str(job), None, None, 0

    def _process_job(self, job: object):
        job_path, ingest_source, ingest_fingerprint, requeue_count = self._parse_job(job)
        logger.info("Processing job: %s", job_path)

        if (
            ingest_source
            and ingest_fingerprint
            and not self._revalidate_queued_fingerprint(
                job_path,
                ingest_source,
                ingest_fingerprint,
                requeue_count=requeue_count,
            )
        ):
            return

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

    def _revalidate_queued_fingerprint(
        self,
        job_path: str,
        ingest_source: str,
        queued_fingerprint: dict[str, int],
        *,
        requeue_count: int,
    ) -> bool:
        source_path = Path(ingest_source)
        current_fingerprint = compute_fingerprint(source_path)
        if current_fingerprint is None:
            logger.warning(
                "Skipping queued job because ingest source disappeared before execution: "
                "job_path=%s ingest_source=%s",
                job_path,
                ingest_source,
            )
            return False

        queued_key = fingerprint_key(queued_fingerprint)
        current_key = fingerprint_key(current_fingerprint)
        if queued_key == current_key:
            return True

        if requeue_count < 1:
            logger.warning(
                "Ingest fingerprint changed after queueing; requeueing once: "
                "job_path=%s ingest_source=%s",
                job_path,
                ingest_source,
            )
            self.job_queue.put((job_path, ingest_source, current_fingerprint, requeue_count + 1))
            return False

        logger.warning(
            "Ingest fingerprint changed again after requeue; skipping stale job: "
            "job_path=%s ingest_source=%s",
            job_path,
            ingest_source,
        )
        return False

    def _record_processed_ingest(
        self,
        ingest_source: str | None,
        ingest_fingerprint: Mapping[str, object] | None,
        *,
        job_path: str | None = None,
    ):
        """Persist ingest completion marker to avoid restart reprocessing loops."""
        if not ingest_source:
            return

        source_path = Path(ingest_source)
        fingerprint = (
            _normalize_fingerprint(ingest_fingerprint)
            if ingest_fingerprint
            else compute_fingerprint(source_path)
        )
        if not fingerprint:
            logger.warning(
                "Unable to fingerprint completed ingest source for dedupe: "
                "ingest_source=%s job_path=%s",
                ingest_source,
                job_path,
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
