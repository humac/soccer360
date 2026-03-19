#!/usr/bin/env bash
# Build YOLO dataset from completed Label Studio exports.
#
# Scans /tank/labeling/ for matches with YOLO-format labels (*.txt),
# creates train/val splits, and writes dataset.yaml.
#
# Usage: scripts/build_dataset.sh [labeling_dir] [val_ratio]
#
# Expects each match directory to have:
#   /tank/labeling/<match>/frames/frame_*.jpg   (images)
#   /tank/labeling/<match>/labels/frame_*.txt   (YOLO labels)

set -euo pipefail

LABELING_DIR="${1:-/tank/labeling}"
VAL_RATIO="${2:-0.2}"
OUTPUT_DIR="${LABELING_DIR}/dataset"

echo "================================================"
echo "Soccer360 Dataset Builder"
echo "  Scanning: $LABELING_DIR"
echo "  Output:   $OUTPUT_DIR"
echo "  Val ratio: $VAL_RATIO"
echo "================================================"

python3 - "$LABELING_DIR" "$VAL_RATIO" "$OUTPUT_DIR" <<'PYEOF'
import json
import random
import re
import shutil
import sys
from pathlib import Path

labeling_dir = Path(sys.argv[1])
val_ratio = float(sys.argv[2])
output_dir = Path(sys.argv[3])
frame_extensions = (".jpg", ".jpeg", ".png")


def candidate_frame_names(label_name: str):
    stem = Path(label_name).stem
    candidates = []
    seen = set()

    def add_stem(candidate_stem: str):
        if not candidate_stem:
            return
        for ext in frame_extensions:
            filename = f"{candidate_stem}{ext}"
            key = filename.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(filename)

    add_stem(stem)
    lower_stem = stem.lower()
    for suffix in ("_jpg", "_jpeg", "_png"):
        if lower_stem.endswith(suffix):
            add_stem(stem[: -len(suffix)])

    frame_match = re.search(r"(frame_\d+)", stem, re.IGNORECASE)
    if frame_match:
        add_stem(frame_match.group(1))

    return candidates


def resolve_frame_for_label(frames_dir: Path, label_name: str):
    for candidate in candidate_frame_names(label_name):
        frame_path = frames_dir / candidate
        if frame_path.is_file():
            return frame_path
    return frames_dir / Path(label_name).with_suffix(".jpg").name

# Collect all image/label pairs from match directories
pairs = []
matches_seen = set()
frame_count = 0
label_count = 0
unmatched_labels = []
for match_dir in sorted(labeling_dir.iterdir()):
    if not match_dir.is_dir() or match_dir.name == "dataset":
        continue

    # Look for YOLO-format labels (exported from Label Studio)
    labels_dir = match_dir / "labels"
    frames_dir = match_dir / "frames"

    if not labels_dir.exists() or not frames_dir.exists():
        continue

    frame_count += sum(1 for path in frames_dir.iterdir() if path.is_file())

    for label_file in sorted(labels_dir.glob("*.txt")):
        if label_file.name.lower() == "classes.txt":
            continue
        label_count += 1
        image_file = resolve_frame_for_label(frames_dir, label_file.name)
        if image_file.exists():
            pairs.append((image_file, label_file, match_dir.name))
            matches_seen.add(match_dir.name)
        elif len(unmatched_labels) < 5:
            unmatched_labels.append(f"{match_dir.name}/{label_file.name}")

if not pairs:
    print("ERROR: No image/label pairs found.")
    print(f"Found {frame_count} frame(s), {label_count} label file(s), {len(pairs)} matched pair(s).")
    if unmatched_labels:
        print(f"Sample unmatched labels: {', '.join(unmatched_labels)}")
    print("")
    print("Expected structure:")
    print("  /tank/labeling/<match>/frames/frame_000123.jpg")
    print("  /tank/labeling/<match>/labels/frame_000123.txt")
    print("")
    print("Steps:")
    print("  1. Process videos to export hard frames automatically")
    print("  2. Label in Label Studio (http://localhost:8080)")
    print("  3. Export labels in YOLO format to /tank/labeling/<match>/labels/")
    sys.exit(1)

print(f"Found {len(pairs)} labeled images across {len(matches_seen)} matches:")
for m in sorted(matches_seen):
    count = sum(1 for p in pairs if p[2] == m)
    print(f"  {m}: {count} images")

# Clean previous dataset
if output_dir.exists():
    shutil.rmtree(str(output_dir))

# Shuffle and split
random.seed(42)
random.shuffle(pairs)
split_idx = max(1, int(len(pairs) * (1 - val_ratio)))
train_pairs = pairs[:split_idx]
val_pairs = pairs[split_idx:]

# Create directory structure and copy files
for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
    img_dir = output_dir / split_name / "images"
    lbl_dir = output_dir / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path, match_name in split_pairs:
        # Prefix with match name to avoid filename collisions
        dest_name = f"{match_name}_{img_path.name}"
        shutil.copy2(str(img_path), str(img_dir / dest_name))
        shutil.copy2(str(lbl_path), str(lbl_dir / dest_name.replace(".jpg", ".txt")))

# Write dataset.yaml
yaml_content = f"""# Soccer360 Ball Detection Dataset
# Auto-generated by build_dataset.sh

path: {output_dir}
train: train/images
val: val/images

nc: 1
names:
  0: ball
"""

dataset_yaml = output_dir / "dataset.yaml"
dataset_yaml.write_text(yaml_content)

print("")
print(f"Dataset built: {len(train_pairs)} train, {len(val_pairs)} val")
print(f"YAML: {dataset_yaml}")
PYEOF
