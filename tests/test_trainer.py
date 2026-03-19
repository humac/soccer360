"""Tests for training model resolution behavior."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

from src.trainer import Trainer


def test_resolve_training_base_model_uses_local_configured_path(tmp_path: Path):
    configured_model = tmp_path / "custom.pt"
    configured_model.write_bytes(b"weights")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    trainer = Trainer({
        "paths": {"models": str(models_dir)},
        "model": {"base_model": str(configured_model)},
    })

    assert trainer._resolve_training_base_model() == configured_model


def test_resolve_training_base_model_falls_back_to_active_model(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    active_model = models_dir / "ball_best.pt"
    active_model.write_bytes(b"weights")

    trainer = Trainer({
        "paths": {"models": str(models_dir)},
        "model": {"base_model": "yolov8m.pt"},
    })

    with patch("src.trainer.Path.is_file", autospec=True) as is_file:
        def _is_file(path_self):
            path = Path(path_self)
            if path == active_model:
                return True
            if path == Path("/app/models/yolo26l.pt"):
                return False
            return path.exists() and path.suffix == ".pt"

        is_file.side_effect = _is_file
        assert trainer._resolve_training_base_model() == active_model


def test_resolve_training_base_model_falls_back_to_baked_default(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    trainer = Trainer({
        "paths": {"models": str(models_dir)},
        "model": {"base_model": "yolov8m.pt"},
    })

    with patch("src.trainer.Path.is_file", autospec=True) as is_file:
        def _is_file(path_self):
            path = Path(path_self)
            if path == Path("/app/models/yolo26l.pt"):
                return True
            return path.exists() and path.suffix == ".pt"

        is_file.side_effect = _is_file
        assert trainer._resolve_training_base_model() == Path("/app/models/yolo26l.pt")


def test_run_uses_zero_dataloader_workers_by_default(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"weights")
    captured: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, model_path: str):
            captured["model_path"] = model_path

        def train(self, **kwargs):
            captured["train_kwargs"] = kwargs
            return {"ok": True}

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    trainer = Trainer({
        "paths": {"models": str(models_dir)},
        "model": {"base_model": str(base_model)},
    })

    result = trainer.run(data=str(tmp_path / "dataset.yaml"), epochs=5)

    assert result == {"ok": True}
    assert captured["model_path"] == str(base_model)
    assert captured["train_kwargs"]["workers"] == 0


def test_run_uses_override_base_model(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    configured_base = tmp_path / "configured.pt"
    configured_base.write_bytes(b"configured")
    override_base = tmp_path / "override.pt"
    override_base.write_bytes(b"override")
    captured: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, model_path: str):
            captured["model_path"] = model_path

        def train(self, **kwargs):
            captured["train_kwargs"] = kwargs
            return {"ok": True}

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    trainer = Trainer({
        "paths": {"models": str(models_dir)},
        "model": {"base_model": str(configured_base)},
    })

    trainer.run(data=str(tmp_path / "dataset.yaml"), epochs=3, base_model=str(override_base))

    assert captured["model_path"] == str(override_base)


def test_run_promotes_named_model_and_updates_active(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"weights")

    class FakeYOLO:
        def __init__(self, model_path: str):
            self.model_path = model_path

        def train(self, **kwargs):
            best_path = models_dir / "ball_model_test" / "weights" / "best.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            best_path.write_bytes(b"trained")
            return {"ok": True}

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    trainer = Trainer({
        "paths": {"models": str(models_dir)},
        "model": {"base_model": str(base_model)},
    })
    monkeypatch.setattr(trainer, "_next_version", lambda: "test")

    trainer.run(
        data=str(tmp_path / "dataset.yaml"),
        epochs=3,
        output_model_name="first_try",
        update_active=True,
    )

    assert (models_dir / "first_try.pt").read_bytes() == b"trained"
    assert (models_dir / "ball_best.pt").read_bytes() == b"trained"
