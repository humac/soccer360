"""Tests for exporter safety guarantees."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.exporter import Exporter
from src.utils import VideoMeta


def _meta() -> VideoMeta:
    return VideoMeta(
        width=640,
        height=320,
        fps=30.0,
        duration=1.0,
        total_frames=30,
        codec="h264",
    )


def _make_work_dir(base: Path, *, with_broadcast: bool = True, with_tactical: bool = True) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    if with_broadcast:
        (base / "broadcast.mp4").write_bytes(b"broadcast")
    if with_tactical:
        (base / "tactical_wide.mp4").write_bytes(b"tactical")

    highlights = base / "highlights"
    highlights.mkdir(exist_ok=True)
    (highlights / "highlight_000.mp4").write_bytes(b"clip")
    return base


def _make_config(tmp_path: Path, *, archive_raw: bool = False, require_tactical: bool = True) -> dict:
    return {
        "paths": {
            "processed": str(tmp_path / "processed"),
            "highlights": str(tmp_path / "highlights"),
            "archive_raw": str(tmp_path / "archive_raw"),
        },
        "exporter": {
            "archive_raw": archive_raw,
            "delete_raw": False,
            "require_tactical": require_tactical,
        },
    }


def test_highlight_rerun_preserves_previous_exports(tmp_path: Path):
    cfg = _make_config(tmp_path)
    exporter = Exporter(cfg)
    meta = _meta()

    raw_input = tmp_path / "ingest" / "game.mp4"
    raw_input.parent.mkdir(parents=True, exist_ok=True)
    raw_input.write_bytes(b"raw")

    work1 = _make_work_dir(tmp_path / "work1")
    out1 = exporter.finalize(work1, str(raw_input), meta)

    work2 = _make_work_dir(tmp_path / "work2")
    out2 = exporter.finalize(work2, str(raw_input), meta)

    assert out1 != out2

    highlights_base = Path(cfg["paths"]["highlights"])
    highlight_dirs = sorted(p for p in highlights_base.iterdir() if p.is_dir())
    assert len(highlight_dirs) == 2
    assert (highlight_dirs[0] / "highlight_000.mp4").exists()
    assert (highlight_dirs[1] / "highlight_000.mp4").exists()


def test_archive_raw_uses_unique_destination(tmp_path: Path):
    cfg = _make_config(tmp_path, archive_raw=True)
    exporter = Exporter(cfg)
    meta = _meta()

    raw_input = tmp_path / "ingest" / "game.mp4"
    raw_input.parent.mkdir(parents=True, exist_ok=True)

    work1 = _make_work_dir(tmp_path / "work1")
    raw_input.write_bytes(b"raw-1")
    exporter.finalize(work1, str(raw_input), meta)

    work2 = _make_work_dir(tmp_path / "work2")
    raw_input.write_bytes(b"raw-2")
    exporter.finalize(work2, str(raw_input), meta)

    archived = sorted((tmp_path / "archive_raw").glob("game*.mp4"))
    assert len(archived) == 2
    assert archived[0].name != archived[1].name


def test_finalize_requires_broadcast_output(tmp_path: Path):
    cfg = _make_config(tmp_path)
    exporter = Exporter(cfg)
    meta = _meta()

    raw_input = tmp_path / "ingest" / "game.mp4"
    raw_input.parent.mkdir(parents=True, exist_ok=True)
    raw_input.write_bytes(b"raw")

    work = _make_work_dir(tmp_path / "work", with_broadcast=False, with_tactical=True)
    with pytest.raises(RuntimeError, match="broadcast.mp4"):
        exporter.finalize(work, str(raw_input), meta)


def test_finalize_requires_tactical_output_by_default(tmp_path: Path):
    cfg = _make_config(tmp_path)
    exporter = Exporter(cfg)
    meta = _meta()

    raw_input = tmp_path / "ingest" / "game.mp4"
    raw_input.parent.mkdir(parents=True, exist_ok=True)
    raw_input.write_bytes(b"raw")

    work = _make_work_dir(tmp_path / "work", with_broadcast=True, with_tactical=False)
    with pytest.raises(RuntimeError, match="tactical_wide.mp4"):
        exporter.finalize(work, str(raw_input), meta)
