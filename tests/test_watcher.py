"""Tests for watcher ingest handling and scratch staging safety."""

from __future__ import annotations

import queue
from pathlib import Path

from watchdog.events import FileMovedEvent

from src.watcher import VideoFileHandler, WatcherDaemon


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

    assert job_queue.get_nowait() == str(expected)
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
