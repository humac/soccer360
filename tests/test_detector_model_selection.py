"""Lightweight resolver tests for ingest model selection."""

from __future__ import annotations

from src.detector import (
    resolve_v1_player_model_path_and_source,
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
        assert source == "runtime.pinned.ball"

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
        assert source == "runtime.auto.ball"

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

    def test_runtime_pinned_player_override_uses_separate_selection_file(self, tmp_path):
        player_model = tmp_path / "player.pt"
        player_model.write_bytes(b"player")

        config = {
            "detector": {
                "player_runtime_override_path": str(tmp_path / "ingest_player_model_selection.json"),
            },
            "mode": {"allow_no_model": True},
        }
        save_v1_runtime_model_selection(config, mode="pinned", path=str(player_model), role="player")

        resolved_path, source = resolve_v1_player_model_path_and_source(config, models_dir=str(tmp_path))
        assert resolved_path == str(player_model)
        assert source == "runtime.pinned.player"

    def test_player_runtime_auto_prefers_base_model_not_ball_best(self, tmp_path):
        fine_tuned_ball = tmp_path / "ball_best.pt"
        fine_tuned_ball.write_bytes(b"ball")
        base_model = tmp_path / "yolo26l.pt"
        base_model.write_bytes(b"player-base")

        config = {
            "detector": {
                "player_runtime_override_path": str(tmp_path / "ingest_player_model_selection.json"),
            },
            "mode": {"allow_no_model": True},
        }
        save_v1_runtime_model_selection(config, mode="auto", role="player")

        resolved_path, source = resolve_v1_player_model_path_and_source(
            config,
            models_dir=str(tmp_path),
            base_model_path=str(base_model),
        )
        assert resolved_path == str(base_model)
        assert source == "runtime.auto.player"
