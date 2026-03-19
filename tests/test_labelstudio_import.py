from __future__ import annotations

import json
from pathlib import Path

from src.labelstudio_import import build_tasks, write_tasks_json


class TestLabelStudioImport:
    def test_build_tasks_supports_active_learning_bbox_manifest(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frame_000002.jpg").write_bytes(b"jpg")

        manifest_path = tmp_path / "hard_frames.json"
        manifest_path.write_text(json.dumps({
            "frames": [
                {"frame_index": 2, "bbox": [192, 96, 384, 192], "conf": 0.35},
            ]
        }))

        tasks = build_tasks("match-a", frames_dir, manifest_path)

        assert len(tasks) == 1
        prediction = tasks[0]["predictions"][0]["result"][0]["value"]
        assert prediction == {
            "x": 10.0,
            "y": 10.0,
            "width": 10.0,
            "height": 10.0,
            "rectanglelabels": ["ball"],
        }

    def test_build_tasks_supports_legacy_predicted_bbox_manifest(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frame_000007.jpg").write_bytes(b"jpg")

        manifest_path = tmp_path / "hard_frames.json"
        manifest_path.write_text(json.dumps({
            "frames": [
                {
                    "frame_index": 7,
                    "predicted_bbox": [96, 48, 288, 144],
                    "predicted_confidence": 0.12,
                },
            ]
        }))

        tasks = build_tasks("match-b", frames_dir, manifest_path)

        assert len(tasks) == 1
        prediction = tasks[0]["predictions"][0]["result"][0]["value"]
        assert prediction["x"] == 5.0
        assert prediction["y"] == 5.0
        assert prediction["width"] == 10.0
        assert prediction["height"] == 10.0
        assert prediction["rectanglelabels"] == ["ball"]

    def test_build_tasks_prefers_predicted_bbox_when_both_shapes_exist(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frame_000003.jpg").write_bytes(b"jpg")

        manifest_path = tmp_path / "hard_frames.json"
        manifest_path.write_text(json.dumps({
            "frames": [
                {
                    "frame_index": 3,
                    "bbox": [0, 0, 192, 96],
                    "predicted_bbox": [192, 96, 384, 192],
                },
            ]
        }))

        tasks = build_tasks("match-c", frames_dir, manifest_path)

        prediction = tasks[0]["predictions"][0]["result"][0]["value"]
        assert prediction["x"] == 10.0
        assert prediction["y"] == 10.0

    def test_build_tasks_emits_task_without_predictions_when_bbox_missing(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frame_000001.jpg").write_bytes(b"jpg")

        manifest_path = tmp_path / "hard_frames.json"
        manifest_path.write_text(json.dumps({
            "frames": [{"frame_index": 1, "triggers": ["lost_run"]}],
        }))

        tasks = build_tasks("match-d", frames_dir, manifest_path)

        assert len(tasks) == 1
        assert "predictions" not in tasks[0]
        assert tasks[0]["data"]["frame_index"] == 1

    def test_write_tasks_json_writes_deterministic_sorted_tasks(self, tmp_path: Path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "frame_000010.jpg").write_bytes(b"jpg")
        (frames_dir / "frame_000002.jpg").write_bytes(b"jpg")

        manifest_path = tmp_path / "hard_frames.json"
        manifest_path.write_text(json.dumps({
            "frames": [
                {"frame_index": 2, "bbox": [192, 96, 384, 192]},
                {"frame_index": 10, "bbox": [0, 0, 192, 96]},
            ]
        }))

        output_dir = tmp_path / "labelstudio"
        output_file = write_tasks_json("match-e", frames_dir, output_dir, manifest_path)
        tasks = json.loads(output_file.read_text())

        assert [task["data"]["frame_index"] for task in tasks] == [2, 10]
