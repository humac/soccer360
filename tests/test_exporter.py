"""Tests for exporter safety guarantees."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

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


def _make_config(
    tmp_path: Path,
    *,
    archive_raw: bool = False,
    require_tactical: bool = True,
) -> dict:
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


# ------------------------------------------------------------------
# Ingest archival tests
# ------------------------------------------------------------------

def _make_ingest_config(
    tmp_path: Path,
    *,
    archive_on_success: bool = True,
    archive_mode: str = "move",
    archive_collision: str = "suffix",
    archive_name_template: str = "{match}_{job_id}{ext}",
) -> dict:
    cfg = _make_config(tmp_path)
    cfg["ingest"] = {
        "archive_on_success": archive_on_success,
        "archive_dir": str(tmp_path / "archive_ingest"),
        "archive_mode": archive_mode,
        "archive_name_template": archive_name_template,
        "archive_collision": archive_collision,
    }
    return cfg


def test_ingest_archive_on_success(tmp_path: Path):
    """Enabled archival moves the original ingest file to the archive dir."""
    cfg = _make_ingest_config(tmp_path, archive_mode="move")
    exporter = Exporter(cfg)

    ingest_file = tmp_path / "ingest" / "game.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"original-video")

    work = _make_work_dir(tmp_path / "work")
    out = exporter.finalize(
        work, str(ingest_file), _meta(),
        ingest_source=str(ingest_file),
        job_id="20250101_120000_game",
    )

    # Source should be gone (moved)
    assert not ingest_file.exists()

    # Archived file should exist
    archive_dir = tmp_path / "archive_ingest"
    archived = list(archive_dir.glob("game_*.mp4"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"original-video"

    # metadata.json should have ingest fields
    meta_path = out / "metadata.json"
    meta = json.loads(meta_path.read_text())
    assert meta["ingest_archive_status"] == "success"
    assert meta["ingest_source_path"] == str(ingest_file)
    assert meta["ingest_archived_path"] is not None
    assert meta["ingest_archive_mode"] == "move"


def test_ingest_archive_disabled_by_default(tmp_path: Path):
    """With archive_on_success=False, ingest file stays in place."""
    cfg = _make_ingest_config(tmp_path, archive_on_success=False)
    exporter = Exporter(cfg)

    ingest_file = tmp_path / "ingest" / "game.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"original-video")

    work = _make_work_dir(tmp_path / "work")
    out = exporter.finalize(
        work, str(ingest_file), _meta(),
        ingest_source=str(ingest_file),
        job_id="20250101_120000_game",
    )

    assert ingest_file.exists()

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["ingest_archive_status"] == "skipped"


def test_ingest_archive_collision_suffix(tmp_path: Path):
    """When destination already exists, a _01 suffix is appended."""
    cfg = _make_ingest_config(tmp_path, archive_collision="suffix")
    exporter = Exporter(cfg)

    # Pre-create a file at the expected destination
    archive_dir = tmp_path / "archive_ingest"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "game_job1.mp4").write_bytes(b"existing")

    ingest_file = tmp_path / "ingest" / "game.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"new-video")

    work = _make_work_dir(tmp_path / "work")
    out = exporter.finalize(
        work, str(ingest_file), _meta(),
        ingest_source=str(ingest_file),
        job_id="job1",
    )

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["ingest_archive_status"] == "success"
    # Should have a suffixed name
    assert "_01" in meta["ingest_archived_path"]

    # Both files should exist
    archived = sorted(archive_dir.glob("game_*.mp4"))
    assert len(archived) == 2


def test_ingest_archive_collision_skip(tmp_path: Path):
    """Skip mode leaves ingest file in place when destination exists."""
    cfg = _make_ingest_config(tmp_path, archive_collision="skip")
    exporter = Exporter(cfg)

    archive_dir = tmp_path / "archive_ingest"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "game_job1.mp4").write_bytes(b"existing")

    ingest_file = tmp_path / "ingest" / "game.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"new-video")

    work = _make_work_dir(tmp_path / "work")
    out = exporter.finalize(
        work, str(ingest_file), _meta(),
        ingest_source=str(ingest_file),
        job_id="job1",
    )

    # Source should still exist (skipped)
    assert ingest_file.exists()

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["ingest_archive_status"] == "skipped"


def test_ingest_archive_collision_overwrite(tmp_path: Path):
    """Overwrite mode replaces existing archive target with new source content."""
    cfg = _make_ingest_config(tmp_path, archive_mode="move", archive_collision="overwrite")
    exporter = Exporter(cfg)

    archive_dir = tmp_path / "archive_ingest"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / "game_job1.mp4"
    target.write_bytes(b"old-video")

    ingest_file = tmp_path / "ingest" / "game.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"new-video")

    work = _make_work_dir(tmp_path / "work")
    out = exporter.finalize(
        work,
        str(ingest_file),
        _meta(),
        ingest_source=str(ingest_file),
        job_id="job1",
    )

    assert not ingest_file.exists()
    assert target.read_bytes() == b"new-video"

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["ingest_archive_status"] == "success"
    assert meta["ingest_archive_collision"] == "overwrite"


def test_ingest_archive_mode_copy(tmp_path: Path):
    """Copy mode preserves the original and places a copy in archive."""
    cfg = _make_ingest_config(tmp_path, archive_mode="copy")
    exporter = Exporter(cfg)

    ingest_file = tmp_path / "ingest" / "game.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"original-video")

    work = _make_work_dir(tmp_path / "work")
    out = exporter.finalize(
        work, str(ingest_file), _meta(),
        ingest_source=str(ingest_file),
        job_id="20250101_120000_game",
    )

    # Source should still exist
    assert ingest_file.exists()

    # Archive should also exist
    archive_dir = tmp_path / "archive_ingest"
    archived = list(archive_dir.glob("game_*.mp4"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == b"original-video"

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["ingest_archive_status"] == "success"


def test_ingest_archive_failure_does_not_raise(tmp_path: Path):
    """Archival failure must not propagate; outputs should be preserved."""
    cfg = _make_ingest_config(tmp_path, archive_mode="move")
    exporter = Exporter(cfg)

    ingest_file = tmp_path / "ingest" / "game.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"original-video")

    work = _make_work_dir(tmp_path / "work")

    with patch.object(Exporter, "_safe_move", side_effect=PermissionError("denied")):
        out = exporter.finalize(
            work, str(ingest_file), _meta(),
            ingest_source=str(ingest_file),
            job_id="20250101_120000_game",
        )

    # Pipeline should still succeed
    assert out.exists()
    assert (out / "broadcast.mp4").exists()

    # Source should remain (move failed)
    assert ingest_file.exists()

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["ingest_archive_status"] == "failed"
    assert meta["ingest_archive_error"] is not None


def test_ingest_archive_template_path_escape_is_blocked(tmp_path: Path):
    """Template must not escape archive_dir via absolute/traversal paths."""
    cfg = _make_ingest_config(
        tmp_path,
        archive_name_template="../outside/{match}_{job_id}{ext}",
    )
    exporter = Exporter(cfg)

    ingest_file = tmp_path / "ingest" / "game.mp4"
    ingest_file.parent.mkdir(parents=True, exist_ok=True)
    ingest_file.write_bytes(b"original-video")

    work = _make_work_dir(tmp_path / "work")
    out = exporter.finalize(
        work,
        str(ingest_file),
        _meta(),
        ingest_source=str(ingest_file),
        job_id="job1",
    )

    # Archival should fail safely and never move outside archive_dir.
    assert ingest_file.exists()
    assert not (tmp_path / "outside").exists()

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["ingest_archive_status"] == "failed"
    assert "escapes archive_dir" in meta["ingest_archive_error"]


def test_ingest_archive_no_source_skipped(tmp_path: Path):
    """When no ingest_source is provided, archival is gracefully skipped."""
    cfg = _make_ingest_config(tmp_path)
    exporter = Exporter(cfg)

    work = _make_work_dir(tmp_path / "work")
    raw_input = tmp_path / "ingest" / "game.mp4"
    raw_input.parent.mkdir(parents=True, exist_ok=True)
    raw_input.write_bytes(b"raw")

    out = exporter.finalize(work, str(raw_input), _meta())

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["ingest_archive_status"] == "skipped"
    assert meta["ingest_source_path"] is None
