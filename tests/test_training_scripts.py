"""Tests for shell training helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DATASET_SCRIPT = REPO_ROOT / "scripts" / "build_dataset.sh"


def test_build_dataset_script_creates_yolo_splits(tmp_path: Path):
    """build_dataset.sh should create train/val splits and dataset.yaml."""
    labeling_dir = tmp_path / "labeling"
    output_dir = labeling_dir / "dataset"

    for match_name in ("match_a", "match_b"):
        frames_dir = labeling_dir / match_name / "frames"
        labels_dir = labeling_dir / match_name / "labels"
        frames_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        for frame_idx in (1, 2):
            stem = f"frame_{frame_idx:06d}"
            (frames_dir / f"{stem}.jpg").write_bytes(b"fake-jpeg-data")
            (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    result = subprocess.run(
        ["bash", str(BUILD_DATASET_SCRIPT), str(labeling_dir), "0.5"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Dataset built: 2 train, 2 val" in result.stdout

    train_images = sorted((output_dir / "train" / "images").glob("*.jpg"))
    train_labels = sorted((output_dir / "train" / "labels").glob("*.txt"))
    val_images = sorted((output_dir / "val" / "images").glob("*.jpg"))
    val_labels = sorted((output_dir / "val" / "labels").glob("*.txt"))

    assert len(train_images) == 2
    assert len(train_labels) == 2
    assert len(val_images) == 2
    assert len(val_labels) == 2

    copied_names = sorted(path.name for path in [*train_images, *val_images])
    assert copied_names == [
        "match_a_frame_000001.jpg",
        "match_a_frame_000002.jpg",
        "match_b_frame_000001.jpg",
        "match_b_frame_000002.jpg",
    ]

    dataset_yaml = output_dir / "dataset.yaml"
    assert dataset_yaml.exists()
    yaml_text = dataset_yaml.read_text()
    assert f"path: {output_dir}" in yaml_text
    assert "train: train/images" in yaml_text
    assert "val: val/images" in yaml_text
    assert "0: ball" in yaml_text


def test_build_dataset_script_matches_common_labelstudio_label_names(tmp_path: Path):
    labeling_dir = tmp_path / "labeling"
    frames_dir = labeling_dir / "match_a" / "frames"
    labels_dir = labeling_dir / "match_a" / "labels"
    frames_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    (frames_dir / "frame_000001.jpg").write_bytes(b"fake-jpeg-data")
    (labels_dir / "frame_000001_jpg.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (labels_dir / "classes.txt").write_text("ball\n")

    result = subprocess.run(
        ["bash", str(BUILD_DATASET_SCRIPT), str(labeling_dir), "0.5"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Dataset built: 1 train, 0 val" in result.stdout or "Dataset built: 0 train, 1 val" in result.stdout
