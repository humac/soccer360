"""Lightweight resolver tests for ingest model selection."""

from __future__ import annotations

from src.detector import (
    resolve_v1_model_path_and_source,
    save_v1_runtime_model_selection,
)


class TestRuntimeIngestModelSelection:
    def test_runtime_pinned_override_beats_detection_path(self, tmp_path):
        detection_model = tmp_path / "legacy_detection.pt"
        detection_model.write_bytes(b"legacy")
        selected_model = tmp_path / "selected.pt"
        selected_model.write_bytes(b"selected")

        config = {
            "detector": {
                "runtime_override_path": str(tmp_path / "ingest_model_selection.json"),
            },
            "detection": {"path": str(detection_model)},
            "mode": {"allow_no_model": True},
        }
        save_v1_runtime_model_selection(config, mode="pinned", path=str(selected_model))

        resolved_path, source = resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))
        assert resolved_path == str(selected_model)
        assert source == "runtime.pinned"

    def test_runtime_auto_prefers_ball_best_when_present(self, tmp_path):
        fine_tuned = tmp_path / "ball_best.pt"
        fine_tuned.write_bytes(b"fine_tuned")
        detection_model = tmp_path / "legacy_detection.pt"
        detection_model.write_bytes(b"legacy")

        config = {
            "detector": {
                "runtime_override_path": str(tmp_path / "ingest_model_selection.json"),
            },
            "detection": {"path": str(detection_model)},
            "mode": {"allow_no_model": True},
        }
        save_v1_runtime_model_selection(config, mode="auto")

        resolved_path, source = resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))
        assert resolved_path == str(fine_tuned)
        assert source == "runtime.auto"

    def test_explicit_detector_override_still_beats_runtime_selection(self, tmp_path):
        detector_model = tmp_path / "detector_override.pt"
        detector_model.write_bytes(b"detector")
        selected_model = tmp_path / "selected.pt"
        selected_model.write_bytes(b"selected")

        config = {
            "detector": {
                "model_path": str(detector_model),
                "runtime_override_path": str(tmp_path / "ingest_model_selection.json"),
            },
            "mode": {"allow_no_model": True},
        }
        save_v1_runtime_model_selection(config, mode="pinned", path=str(selected_model))

        resolved_path, source = resolve_v1_model_path_and_source(config, models_dir=str(tmp_path))
        assert resolved_path == str(detector_model)
        assert source == "detector.model_path"
