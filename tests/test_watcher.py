"""Tests for watcher ingest handling and scratch staging safety."""

from __future__ import annotations

import os
import queue
import re
import time
from pathlib import Path
from types import SimpleNamespace

from watchdog.events import FileMovedEvent

from src.watcher import (
    ProcessedIngestStore,
    VideoFileHandler,
    WatcherDaemon,
    _file_fingerprint,
    compute_fingerprint,
)


class _ImmediateFuture:
    def result(self):
        return None


def _patch_immediate_submit(handler: VideoFileHandler, monkeypatch):
    def submit(fn, *args, **kwargs):
        fn(*args, **kwargs)
        return _ImmediateFuture()

    monkeypatch.setattr(handler._executor, "submit", submit)


def test_on_moved_queues_video_job(tmp_path: Path, monkeypatch):
    job_queue: queue.Queue = queue.Queue()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    handler = VideoFileHandler(
        job_queue,
        scratch_dir,
        stability_checks=1,
        stability_interval=0.0,
        max_copy_workers=4,
    )
    _patch_immediate_submit(handler, monkeypatch)
    monkeypatch.setattr(handler, "_wait_stable", lambda _: True)

    expected = scratch_dir / "job_001" / "match.mp4"

    def fake_copy(path: Path) -> Path:
        expected.parent.mkdir(parents=True, exist_ok=True)
        expected.write_bytes(path.read_bytes())
        return expected

    monkeypatch.setattr(handler, "_copy_to_scratch", fake_copy)

    src = tmp_path / "match.uploading"
    src.write_bytes(b"partial")
    dst = tmp_path / "match.mp4"
    dst.write_bytes(b"video")
    handler.on_moved(FileMovedEvent(str(src), str(dst)))

    job_path, ingest_source, ingest_fingerprint = job_queue.get_nowait()
    assert job_path == str(expected)
    assert ingest_source == str(dst)
    assert ingest_fingerprint["size"] == len(b"video")
    handler.close()


def test_copy_to_scratch_creates_unique_job_dirs(tmp_path: Path):
    job_queue: queue.Queue = queue.Queue()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    source = tmp_path / "game.mp4"
    source.write_bytes(b"abc123")

    handler = VideoFileHandler(job_queue, scratch_dir, max_copy_workers=4)
    staged1 = handler._copy_to_scratch(source)
    staged2 = handler._copy_to_scratch(source)

    assert staged1.parent != staged2.parent
    assert staged1.exists()
    assert staged2.exists()
    handler.close()


def test_dotfile_ignored(tmp_path: Path):
    """Hidden files (dotfiles) should be silently ignored."""
    job_queue: queue.Queue = queue.Queue()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    handler = VideoFileHandler(job_queue, scratch_dir, max_copy_workers=1)

    # Create various dotfiles
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / ".nfs000001").write_bytes(b"x")
    (tmp_path / ".hidden_video.mp4").write_bytes(b"x")

    handler._handle_candidate(tmp_path / ".DS_Store")
    handler._handle_candidate(tmp_path / ".nfs000001")
    handler._handle_candidate(tmp_path / ".hidden_video.mp4")

    assert job_queue.empty()
    handler.close()


def test_part_file_ignored(tmp_path: Path):
    """Files with staging suffixes (.part, .tmp) should be ignored."""
    job_queue: queue.Queue = queue.Queue()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    handler = VideoFileHandler(job_queue, scratch_dir, max_copy_workers=1)

    (tmp_path / "match.part").write_bytes(b"partial")
    (tmp_path / "match.tmp").write_bytes(b"partial")

    handler._handle_candidate(tmp_path / "match.part")
    handler._handle_candidate(tmp_path / "match.tmp")

    assert job_queue.empty()
    handler.close()


def test_duplicate_events_are_deduped_while_in_flight(tmp_path: Path, monkeypatch):
    """A second event for the same path should be ignored while first is in progress."""
    job_queue: queue.Queue = queue.Queue()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    source = tmp_path / "match.mp4"
    source.write_bytes(b"video")

    handler = VideoFileHandler(job_queue, scratch_dir, max_copy_workers=1)
    submissions: list[tuple] = []

    def submit(fn, *args, **kwargs):
        submissions.append((fn, args, kwargs))
        return _ImmediateFuture()

    monkeypatch.setattr(handler._executor, "submit", submit)

    handler._handle_candidate(source)
    handler._handle_candidate(source)

    assert len(submissions) == 1
    handler.close()


def test_processed_state_skips_already_processed_file(tmp_path: Path, monkeypatch):
    """Persisted state should prevent startup/event reprocessing loops."""
    job_queue: queue.Queue = queue.Queue()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    source = tmp_path / "match.mp4"
    source.write_bytes(b"video")

    state = ProcessedIngestStore(scratch_dir / "processed_state.json")
    fingerprint = _file_fingerprint(source)
    assert fingerprint is not None
    state.mark_processed(source, fingerprint, job_path="job1/match.mp4")

    handler = VideoFileHandler(
        job_queue,
        scratch_dir,
        max_copy_workers=1,
        processed_store=state,
    )
    submissions: list[tuple] = []

    def submit(fn, *args, **kwargs):
        submissions.append((fn, args, kwargs))
        return _ImmediateFuture()

    monkeypatch.setattr(handler._executor, "submit", submit)
    handler._handle_candidate(source)

    assert len(submissions) == 0
    assert job_queue.empty()
    handler.close()


def test_replaced_file_is_reprocessed_after_success_marker(tmp_path: Path, monkeypatch):
    """Changed file at same path must not be skipped by persisted dedupe marker."""
    job_queue: queue.Queue = queue.Queue()
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    source = tmp_path / "match.mp4"
    source.write_bytes(b"video-v1")

    state = ProcessedIngestStore(scratch_dir / "processed_state.json")
    first_fp = _file_fingerprint(source)
    assert first_fp is not None
    state.mark_processed(source, first_fp, job_path="job1/match.mp4")

    # Replace content to force a new fingerprint for same path.
    source.write_bytes(b"video-v2-with-different-size")

    handler = VideoFileHandler(
        job_queue,
        scratch_dir,
        max_copy_workers=1,
        processed_store=state,
    )
    submissions: list[tuple] = []

    def submit(fn, *args, **kwargs):
        submissions.append((fn, args, kwargs))
        return _ImmediateFuture()

    monkeypatch.setattr(handler._executor, "submit", submit)
    handler._handle_candidate(source)

    assert len(submissions) == 1
    handler.close()


def test_record_processed_ingest_marks_state(test_config, tmp_path: Path):
    """Successful job completion should persist ingest dedupe marker."""
    cfg = {
        **test_config,
        "paths": {
            **test_config["paths"],
            "ingest": str(tmp_path / "ingest"),
            "scratch": str(tmp_path / "scratch"),
        },
    }
    daemon = WatcherDaemon(cfg)
    ingest_file = tmp_path / "ingest" / "match.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"video")

    fingerprint = _file_fingerprint(ingest_file)
    assert fingerprint is not None
    daemon._record_processed_ingest(
        str(ingest_file),
        fingerprint,
        job_path="job_123/match.mp4",
    )

    reloaded = ProcessedIngestStore(daemon.processed_store.state_file)
    assert reloaded.is_processed(ingest_file, fingerprint)


def test_create_handler_uses_configured_limits(test_config):
    cfg = {
        **test_config,
        "watcher": {
            "stability_checks": 7,
            "stability_interval_sec": 0.25,
            "ignore_suffixes": [".uploading", ".tmp", ".part", ".ignore"],
            "max_copy_workers": 4,
        },
    }
    daemon = WatcherDaemon(cfg)
    handler = daemon._create_handler()

    assert handler.stability_checks == 7
    assert abs(handler.stability_interval - 0.25) < 1e-9
    assert ".ignore" in handler.ignore_suffixes
    assert handler._executor._max_workers == 4
    handler.close()


def test_startup_logs_processed_state_status_once(test_config, tmp_path: Path, caplog):
    cfg = {
        **test_config,
        "paths": {
            **test_config["paths"],
            "processed": str(tmp_path / "processed"),
            "scratch": str(tmp_path / "scratch"),
        },
    }

    caplog.set_level("INFO", logger="soccer360.watcher")
    daemon = WatcherDaemon(cfg)

    pattern = re.compile(
        r"ingest_dedupe_state .*processed_state_file=.* persistence=enabled"
    )
    matches = [r for r in caplog.records if pattern.search(r.getMessage())]
    assert len(matches) == 1
    assert str(daemon.processed_store.state_file) in matches[0].getMessage()


def test_startup_logs_degraded_when_state_dir_unwritable(
    test_config, tmp_path: Path, monkeypatch, caplog
):
    cfg = {
        **test_config,
        "paths": {
            **test_config["paths"],
            "processed": str(tmp_path / "processed"),
            "scratch": str(tmp_path / "scratch"),
            "ingest": str(tmp_path / "ingest"),
        },
        "watcher": {
            **test_config["watcher"],
        },
    }
    cfg["watcher"].pop("processed_state_file", None)

    state_dir = Path(cfg["paths"]["processed"]) / ".state"
    original_mkdir = Path.mkdir

    def fail_state_dir_mkdir(self: Path, *args, **kwargs):
        if self == state_dir:
            raise PermissionError("read-only state dir")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_state_dir_mkdir, raising=True)

    caplog.set_level("INFO", logger="soccer360.watcher")
    daemon = WatcherDaemon(cfg)

    pattern = re.compile(
        r"ingest_dedupe_state .*processed_state_file=.* persistence=degraded reason="
    )
    matches = [r for r in caplog.records if pattern.search(r.getMessage())]
    assert len(matches) == 1

    ingest_file = Path(cfg["paths"]["ingest"]) / "match.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"video")
    fp = _file_fingerprint(ingest_file)
    assert fp is not None

    # Degraded persistence must not crash normal processing flow.
    daemon._record_processed_ingest(str(ingest_file), fp, job_path="job/match.mp4")
    assert not daemon.processed_store.state_file.exists()


def test_processed_state_default_is_under_processed_root(test_config, tmp_path: Path):
    cfg = {
        **test_config,
        "paths": {
            **test_config["paths"],
            "processed": str(tmp_path / "processed"),
            "scratch": str(tmp_path / "scratch"),
        },
        "watcher": {
            **test_config["watcher"],
        },
    }
    cfg["watcher"].pop("processed_state_file", None)

    daemon = WatcherDaemon(cfg)
    assert daemon.processed_store.state_file == (
        Path(cfg["paths"]["processed"]) / ".state" / "watcher_processed_ingest.json"
    )


def test_processed_state_relative_path_resolves_under_processed_root(
    test_config, tmp_path: Path
):
    cfg = {
        **test_config,
        "paths": {
            **test_config["paths"],
            "processed": str(tmp_path / "processed"),
            "scratch": str(tmp_path / "scratch"),
        },
        "watcher": {
            **test_config["watcher"],
            "processed_state_file": "custom/subdir/state.json",
        },
    }
    daemon = WatcherDaemon(cfg)
    assert daemon.processed_store.state_file == (
        Path(cfg["paths"]["processed"]) / ".state" / "custom" / "subdir" / "state.json"
    )


def test_processed_state_atomic_publish_uses_replace(tmp_path: Path, monkeypatch):
    state_file = tmp_path / "state.json"
    store = ProcessedIngestStore(state_file)
    source = tmp_path / "match.mp4"
    source.write_bytes(b"video")
    fp = _file_fingerprint(source)
    assert fp is not None

    replace_calls: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def tracking_replace(src: str | os.PathLike, dst: str | os.PathLike):
        replace_calls.append((Path(src), Path(dst)))
        return original_replace(src, dst)

    monkeypatch.setattr("src.watcher.os.replace", tracking_replace)
    store.mark_processed(source, fp, job_path="job/match.mp4")

    assert any(
        dst == state_file
        and src.parent == state_file.parent
        and src.name.startswith(f".{state_file.name}.tmp-")
        for src, dst in replace_calls
    )


def test_processed_state_corruption_is_quarantined(tmp_path: Path):
    state_file = tmp_path / "processed_state.json"
    state_file.write_text("{not valid json", encoding="utf-8")

    store = ProcessedIngestStore(state_file)
    assert store._entries == {}
    assert not state_file.exists()

    quarantined = list(tmp_path.glob("processed_state.json.corrupt.*"))
    assert len(quarantined) == 1

    source = tmp_path / "match.mp4"
    source.write_bytes(b"video")
    fp = _file_fingerprint(source)
    assert fp is not None
    store.mark_processed(source, fp, job_path="job/match.mp4")

    reloaded = ProcessedIngestStore(state_file)
    assert reloaded.is_processed(source, fp)


def test_processed_state_prunes_to_max_entries(tmp_path: Path):
    state_file = tmp_path / "processed_state.json"
    store = ProcessedIngestStore(state_file, max_entries=2)

    files = []
    for idx in range(3):
        source = tmp_path / f"match_{idx}.mp4"
        source.write_bytes(f"video-{idx}".encode("utf-8"))
        fp = _file_fingerprint(source)
        assert fp is not None
        store.mark_processed(source, fp, job_path=f"job{idx}/match.mp4")
        files.append((source, fp))
        time.sleep(0.002)

    reloaded = ProcessedIngestStore(state_file, max_entries=2)
    assert len(reloaded._entries) == 2
    assert not reloaded.is_processed(files[0][0], files[0][1])
    assert reloaded.is_processed(files[1][0], files[1][1])
    assert reloaded.is_processed(files[2][0], files[2][1])


def test_fingerprint_ignores_zero_dev_ino(tmp_path: Path, monkeypatch):
    source = tmp_path / "match.mp4"
    source.write_bytes(b"video")

    original_stat = Path.stat

    def fake_stat(self: Path, *args, **kwargs):
        if self == source:
            return SimpleNamespace(
                st_size=123,
                st_mtime_ns=456,
                st_ctime_ns=789,
                st_ino=0,
                st_dev=0,
            )
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat, raising=True)

    fp = compute_fingerprint(source)
    assert fp is not None
    assert fp["size"] == 123
    assert fp["mtime_ns"] == 456
    assert fp["ctime_ns"] == 789
    assert "ino" not in fp
    assert "dev" not in fp

    state = ProcessedIngestStore(tmp_path / "state.json")
    state.mark_processed(source, fp, job_path="job/match.mp4")
    assert state.is_processed(source, fp)


def test_process_job_revalidates_fingerprint_and_requeues_once(
    test_config, tmp_path: Path, monkeypatch
):
    cfg = {
        **test_config,
        "paths": {
            **test_config["paths"],
            "ingest": str(tmp_path / "ingest"),
            "scratch": str(tmp_path / "scratch"),
            "processed": str(tmp_path / "processed"),
        },
    }
    daemon = WatcherDaemon(cfg)

    ingest_file = tmp_path / "ingest" / "match.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"video-v1")
    queued_fp = _file_fingerprint(ingest_file)
    assert queued_fp is not None

    ingest_file.write_bytes(b"video-v2-updated")

    pipeline_calls: list[tuple] = []

    class _FakePipeline:
        def __init__(self, _cfg: dict):
            self.cfg = _cfg

        def run(self, *args, **kwargs):
            pipeline_calls.append((args, kwargs))

    monkeypatch.setattr("src.watcher.Pipeline", _FakePipeline)

    requeued: list[tuple] = []
    monkeypatch.setattr(daemon.job_queue, "put", lambda item: requeued.append(item))

    daemon._process_job(("job_1/match.mp4", str(ingest_file), queued_fp, 0))
    assert not pipeline_calls
    assert len(requeued) == 1
    assert requeued[0][3] == 1

    ingest_file.write_bytes(b"video-v3-updated-again")
    daemon._process_job(requeued[0])
    assert not pipeline_calls
    assert len(requeued) == 1
