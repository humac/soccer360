# Roboflow Sports Ball Dataset -- Migration SOP

Standard operating procedure for downloading a Roboflow sports ball dataset and integrating it into Soccer360's YOLO training pipeline to improve ball detection.

---

## Table of Contents

- [Background](#background)
- [Prerequisites](#prerequisites)
- [Step 1: Create a Roboflow Account](#step-1-create-a-roboflow-account)
- [Step 2: Download the Dataset](#step-2-download-the-dataset)
- [Step 3: Understand the Class ID Mismatch](#step-3-understand-the-class-id-mismatch)
- [Step 4: Remap Class IDs](#step-4-remap-class-ids)
- [Step 5: Verify Remapped Labels](#step-5-verify-remapped-labels)
- [Step 6: Merge into Soccer360 Dataset](#step-6-merge-into-soccer360-dataset)
- [Step 7: Rebuild Dataset and Retrain](#step-7-rebuild-dataset-and-retrain)
- [Step 8: Evaluate the New Model](#step-8-evaluate-the-new-model)
- [Appendix: Why Not Use Roboflow Weights Directly?](#appendix-why-not-use-roboflow-weights-directly)
- [Appendix: Adapting for Other Sports](#appendix-adapting-for-other-sports)

---

## Background

Soccer360's ball detection uses a YOLO model fine-tuned from `yolo26l.pt` (COCO-pretrained). The pipeline's active learning loop exports hard frames (low-confidence, lost-ball, jump events) for human labeling in Label Studio, then retrains. This works but starts from a small dataset.

Roboflow hosts publicly available sports ball datasets with thousands of pre-annotated images from professional broadcast footage. Merging this data with your existing labels gives the model significantly more training examples without additional manual labeling effort.

**Recommended dataset:** Roboflow "football-players-detection" project

- 4,377+ annotated images from professional soccer footage
- 4 classes: ball, player, referee, goalkeeper
- Reported mAP@0.50 of 0.925
- Available in YOLO format

---

## Prerequisites

- Roboflow account (free tier is sufficient for dataset download)
- Python 3 with pip available on the host
- Access to `/tank/labeling/` directory on the Soccer360 server
- A working Soccer360 installation with dashboard access

---

## Step 1: Create a Roboflow Account

1. Go to https://universe.roboflow.com
2. Sign up for a free account
3. Navigate to the football-players-detection project: https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
4. Note: you can browse the dataset images and annotations before downloading

---

## Step 2: Download the Dataset

Option A -- using the Roboflow Python SDK:

```bash
pip install roboflow

python3 -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')
project = rf.workspace('roboflow-jvuqo').project('football-players-detection-3zvbc')
version = project.version(1)
dataset = version.download('yolov8', location='/tank/labeling/roboflow_download')
"
```

Option B -- manual download from the Roboflow web UI:

1. Open the dataset version page
2. Click "Download Dataset"
3. Select format: **YOLOv8**
4. Download and extract to `/tank/labeling/roboflow_download/`

After download, the directory structure looks like:

```
/tank/labeling/roboflow_download/
  train/
    images/
      img001.jpg
      img002.jpg
      ...
    labels/
      img001.txt
      img002.txt
      ...
  valid/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
```

---

## Step 3: Understand the Class ID Mismatch

This is the critical step. **You cannot use Roboflow weights or labels as-is.**

| Class       | Roboflow ID | Soccer360 (COCO) ID |
|-------------|-------------|----------------------|
| Ball        | 0           | 32                   |
| Player      | 1           | 0                    |
| Referee     | 2           | (not used)           |
| Goalkeeper  | 3           | (not used)           |

Soccer360's pipeline is hardwired to these COCO class IDs:

- **Class 32** (ball) -- used by `Detector`, `BallStabilizer`, `Tracker`, camera path, highlights
- **Class 0** (person) -- used by `PlayerClusterComputer` for center-of-play tracking

If you feed Roboflow labels without remapping, the pipeline would treat balls as players and players as an unknown class. Detection would fail silently.

---

## Step 4: Remap Class IDs

Run the remapping script below. It converts Roboflow label files so that:

- Roboflow class 0 (ball) becomes Soccer360 class 0 (ball in training convention)
- All other classes (player, referee, goalkeeper) are discarded

> **Why class 0 and not class 32?** Soccer360's training dataset uses a single-class convention where class 0 = ball. The dataset builder handles the mapping to COCO class 32 at training time. This matches how Label Studio exports work (see the labeling guide).

```bash
python3 - <<'REMAP_SCRIPT'
"""
Remap Roboflow football-players-detection labels for Soccer360.

Reads YOLO-format .txt label files, keeps only ball annotations
(Roboflow class 0), remaps them to Soccer360 training class 0,
and writes to an output directory.
"""
import os
import shutil
from pathlib import Path

ROBOFLOW_DIR = Path("/tank/labeling/roboflow_download")
OUTPUT_DIR   = Path("/tank/labeling/roboflow_remapped")

# Roboflow class 0 = ball. We keep only ball annotations.
# Output class ID = 0 (Soccer360 single-class training convention).
ROBOFLOW_BALL_CLASS = 0
OUTPUT_BALL_CLASS   = 0

for split in ("train", "valid", "test"):
    src_images = ROBOFLOW_DIR / split / "images"
    src_labels = ROBOFLOW_DIR / split / "labels"
    dst_images = OUTPUT_DIR / split / "images"
    dst_labels = OUTPUT_DIR / split / "labels"

    if not src_labels.is_dir():
        print(f"  Skipping {split}/ (not found)")
        continue

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0

    for label_file in sorted(src_labels.glob("*.txt")):
        lines_out = []
        for line in label_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id == ROBOFLOW_BALL_CLASS:
                # Remap to output class and keep bbox coordinates as-is
                lines_out.append(f"{OUTPUT_BALL_CLASS} {' '.join(parts[1:])}")

        if not lines_out:
            skipped += 1
            continue

        # Write remapped label
        (dst_labels / label_file.name).write_text("\n".join(lines_out) + "\n")

        # Copy corresponding image
        stem = label_file.stem
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            src_img = src_images / (stem + ext)
            if src_img.exists():
                shutil.copy2(src_img, dst_images / src_img.name)
                break

        kept += 1

    print(f"  {split}: kept {kept} images with ball annotations, skipped {skipped} without ball")

print(f"\nRemapped dataset written to: {OUTPUT_DIR}")
print("Only ball annotations retained. All player/referee/goalkeeper annotations discarded.")
REMAP_SCRIPT
```

Expected output:

```
  train: kept ~3500 images with ball annotations, skipped ~100 without ball
  valid: kept ~400 images with ball annotations, skipped ~20 without ball
  test: kept ~200 images with ball annotations, skipped ~10 without ball

Remapped dataset written to: /tank/labeling/roboflow_remapped
Only ball annotations retained. All player/referee/goalkeeper annotations discarded.
```

---

## Step 5: Verify Remapped Labels

Spot-check a few label files to confirm the format is correct:

```bash
# Should show lines like: 0 0.523 0.341 0.015 0.028
head -3 /tank/labeling/roboflow_remapped/train/labels/*.txt | head -20

# Count total annotations
wc -l /tank/labeling/roboflow_remapped/train/labels/*.txt | tail -1

# Verify all class IDs are 0
awk '{print $1}' /tank/labeling/roboflow_remapped/train/labels/*.txt | sort -u
# Expected output: 0
```

Each label line must have exactly 5 values: `class_id x_center y_center width height`, all normalized 0.0-1.0.

---

## Step 6: Merge into Soccer360 Dataset

Copy the remapped Roboflow images and labels into your existing labeling directory structure so they are picked up by the dataset builder.

```bash
# Create a pseudo-match directory for the Roboflow data
ROBOFLOW_MATCH="/tank/labeling/roboflow_football"
mkdir -p "$ROBOFLOW_MATCH/frames"
mkdir -p "$ROBOFLOW_MATCH/labels"

# Copy remapped train + valid images and labels
# (test split is held out -- do not include in training)
cp /tank/labeling/roboflow_remapped/train/images/* "$ROBOFLOW_MATCH/frames/"
cp /tank/labeling/roboflow_remapped/train/labels/* "$ROBOFLOW_MATCH/labels/"
cp /tank/labeling/roboflow_remapped/valid/images/* "$ROBOFLOW_MATCH/frames/"
cp /tank/labeling/roboflow_remapped/valid/labels/* "$ROBOFLOW_MATCH/labels/"

# Verify counts
echo "Frames: $(ls "$ROBOFLOW_MATCH/frames/" | wc -l)"
echo "Labels: $(ls "$ROBOFLOW_MATCH/labels/" | wc -l)"
```

The directory `/tank/labeling/roboflow_football/` now appears alongside your existing match directories (e.g., `match_2025-01-15_game1/`). The dataset builder will include it automatically.

---

## Step 7: Rebuild Dataset and Retrain

### Option A: Dashboard (recommended)

1. Open Soccer360 dashboard at `http://<server>:8088`
2. Go to the **Training** section
3. Click **Build Dataset** -- this scans all match directories under `/tank/labeling/` including the new `roboflow_football/` directory
4. Verify the build summary shows increased image/label counts
5. Click **Train** to start fine-tuning from `yolo26l.pt`

### Option B: Command line

```bash
# Build dataset
docker compose run --rm --entrypoint python worker -c "
from src.dashboard import _build_dataset_from_labels
from pathlib import Path
result = _build_dataset_from_labels(labeling_dir=Path('/tank/labeling'))
print(result)
"

# Train (uses the dataset built above)
docker compose run --rm worker train \
  --data /tank/labeling/dataset/dataset.yaml \
  --base-model /app/models/yolo26l.pt \
  --epochs 100
```

---

## Step 8: Evaluate the New Model

After training completes:

1. The new best weights are saved to `/tank/models/ball_best.pt`
2. Run a test ingest on a match you know has ball visibility issues
3. Compare broadcast output quality against the previous model
4. Check detection metrics in the dashboard's Detection Settings page

### Keep the Roboflow test split for validation

```bash
# Run inference on held-out Roboflow test images
docker compose run --rm --entrypoint python worker -c "
from ultralytics import YOLO
model = YOLO('/tank/models/ball_best.pt')
results = model.val(data='/tank/labeling/roboflow_remapped/test/')
print(f'mAP@50: {results.box.map50:.3f}')
print(f'mAP@50-95: {results.box.map:.3f}')
"
```

---

## Appendix: Why Not Use Roboflow Weights Directly?

Three reasons:

1. **Class ID mismatch** -- Roboflow uses custom IDs (0=ball, 1=player). Soccer360 uses COCO IDs (32=ball, 0=person). The pipeline filters, tracks, and routes detections by class ID throughout `detector.py`, `tracker.py`, `player_cluster.py`, `highlights.py`, and `camera.py`. Using mismatched weights would silently produce wrong results.

2. **Architecture mismatch** -- The Roboflow model may be YOLOv8/v9/v11. Soccer360 uses YOLO26l. Weights from a different architecture cannot be loaded.

3. **Domain gap** -- Roboflow data is standard broadcast 2D footage. Soccer360 processes 360-degree equirectangular video with different distortion characteristics. Fine-tuning on your own 360 hard frames combined with Roboflow's broadcast data produces a model that handles both domains.

The correct approach is to use Roboflow's **annotations** (not weights) as additional training data for your existing pipeline.

---

## Appendix: Adapting for Other Sports

The same workflow applies to other sports. Find a suitable Roboflow dataset, remap class IDs, and merge.

| Sport      | Roboflow Search Terms            | Ball Class to Keep |
|------------|----------------------------------|--------------------|
| Soccer     | football-players-detection       | ball (class 0)     |
| Hockey     | hockey puck detection            | puck               |
| Basketball | basketball detection             | basketball         |
| Tennis     | tennis ball detection            | ball               |

For each sport:

1. Search https://universe.roboflow.com for the relevant dataset
2. Download in YOLOv8 format
3. Inspect `data.yaml` to identify which class ID corresponds to the ball/puck
4. Modify the `ROBOFLOW_BALL_CLASS` variable in the remap script (Step 4)
5. Follow the same merge and retrain workflow

The rest of the Soccer360 pipeline (tracking, camera path, highlights) works sport-agnostically -- it tracks a ball-class object regardless of the sport. Only the detection model needs sport-specific training data.
