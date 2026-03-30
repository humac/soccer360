# Soccer360 -- Label Studio Labeling Guide

A step-by-step guide for using Label Studio to annotate hard frames and improve the Soccer360 ball detection model. This guide covers the complete workflow from opening Label Studio through exporting labels and training an improved model.

---

## Table of Contents

- [Overview](#overview)
- [Before You Begin](#before-you-begin)
- [Part 1: First-Time Setup](#part-1-first-time-setup)
  - [1.1 Opening Label Studio](#11-opening-label-studio)
  - [1.2 Creating Your Account](#12-creating-your-account)
  - [1.3 Understanding the Interface](#13-understanding-the-interface)
- [Part 2: Importing Hard Frames](#part-2-importing-hard-frames)
  - [2.1 What Are Hard Frames?](#21-what-are-hard-frames)
  - [2.2 Generating Import Tasks](#22-generating-import-tasks)
  - [2.3 Creating a Labeling Project](#23-creating-a-labeling-project)
  - [2.4 Configuring the Labeling Interface](#24-configuring-the-labeling-interface)
  - [2.5 Importing Tasks Into the Project](#25-importing-tasks-into-the-project)
- [Part 3: Labeling Ball Positions](#part-3-labeling-ball-positions)
  - [3.1 Opening a Task](#31-opening-a-task)
  - [3.2 Drawing a Bounding Box](#32-drawing-a-bounding-box)
  - [3.3 Using Pre-Annotations](#33-using-pre-annotations)
  - [3.4 Handling Difficult Frames](#34-handling-difficult-frames)
  - [3.5 Keyboard Shortcuts](#35-keyboard-shortcuts)
  - [3.6 Tips for Accurate Labels](#36-tips-for-accurate-labels)
- [Part 4: Exporting Labels](#part-4-exporting-labels)
  - [4.1 Exporting in YOLO Format](#41-exporting-in-yolo-format)
  - [4.2 Uploading Labels via the Dashboard (Recommended)](#42-uploading-labels-via-the-dashboard-recommended)
  - [4.3 Placing Labels Manually (Alternative)](#43-placing-labels-manually-alternative)
- [Part 5: Building the Dataset and Training](#part-5-building-the-dataset-and-training)
  - [5.1 Using the Dashboard (Recommended)](#51-using-the-dashboard-recommended)
  - [5.2 Using the Command Line](#52-using-the-command-line)
  - [5.3 Verifying the New Model](#53-verifying-the-new-model)
- [Part 6: The Weekly Improvement Cycle](#part-6-the-weekly-improvement-cycle)
- [Reference](#reference)
  - [Directory Structure](#directory-structure)
  - [Hard Frame Trigger Types](#hard-frame-trigger-types)
  - [Label Format (YOLO)](#label-format-yolo)
  - [Troubleshooting](#troubleshooting)

---

## Overview

Soccer360 automatically identifies frames where the ball detection model struggled -- these are called **hard frames**. By labeling the ball position in these frames and retraining, the model improves over time. This is the active learning loop:

```text
Process videos --> Hard frames exported --> Label in Label Studio --> Build dataset --> Train model --> Better results
```

You don't need to be a machine learning expert. The labeling task is straightforward: draw a box around the ball in each image.

> **Note:** The pipeline now detects both players and the ball during processing, but labeling and training focus exclusively on ball detection. Player detection uses the pretrained model and does not need manual labeling.

## Before You Begin

Make sure the following are running:

```bash
docker compose ps
```

You should see:

| Service | Status | Port |
| --- | --- | --- |
| soccer360-worker | Up (healthy) | -- |
| soccer360-labelstudio | Up (healthy) | 8080 |
| soccer360-dashboard | Up | 8088 |

If Label Studio is not running:

```bash
docker compose up -d labelstudio
```

You also need at least one processed match with exported hard frames. Check:

```bash
ls /tank/labeling/
```

Each listed directory is a match with hard frames ready for labeling.

---

## Part 1: First-Time Setup

### 1.1 Opening Label Studio

Open your web browser and navigate to:

```text
http://<server-address>:8080
```

Replace `<server-address>` with your server's IP address or hostname. This is the same host where the dashboard runs on port 8088.

> **Tip:** If you're already on the Soccer360 dashboard, click the **Open Label Studio** button in the Active Learning section -- it automatically uses the correct address.

### 1.2 Creating Your Account

The first time you access Label Studio, you'll see a sign-up page.

1. Enter your **email address** (this is your login, it doesn't need to be a real email)
2. Choose a **password**
3. Click **Sign Up**

> **Note:** This is a local account on your server. There is no email verification. Label Studio stores its data in a Docker volume, so your account persists across restarts.

After signing up, you'll see the main **Projects** page.

### 1.3 Understanding the Interface

The Label Studio interface has three main areas:

- **Projects page** -- lists all your labeling projects (one per match is recommended)
- **Data Manager** -- shows all imported frames for a project, with completion status
- **Labeling view** -- where you draw bounding boxes on individual frames

---

## Part 2: Importing Hard Frames

### 2.1 What Are Hard Frames?

During pipeline processing, the system identifies frames where the model had difficulty:

| Trigger | What It Means | Why It Helps |
| --- | --- | --- |
| **Low confidence** | Model detected a ball but wasn't sure (confidence 10-50%) | Teaching the model to be more certain |
| **Lost ball run** | Ball wasn't detected for 5+ consecutive frames | Teaching the model to find the ball in new situations |
| **Jump rejection** | Ball position jumped impossibly far between frames | Teaching the model to avoid false detections |

Hard frames are automatically saved to `/tank/labeling/<match_name>/frames/` during processing.

### 2.2 Generating Import Tasks

You can generate task files from either the dashboard or the command line.

**From the Dashboard (recommended):**

1. Open the Soccer360 Dashboard at `http://<server-address>:8088`
2. Switch to the **Labeling & Training** workspace tab
3. Find your match in the list
4. Click the **Import** button next to the match name
5. A toast notification confirms the number of tasks created

**From the command line:**

```bash
bash scripts/labelstudio_import.sh <match_name>
```

For example:

```bash
bash scripts/labelstudio_import.sh LastGame-Test
```

Both methods create `/tank/labeling/<match_name>/labelstudio/tasks.json` containing image references for each hard frame and pre-annotations where bounding-box metadata is available in `hard_frames.json`.

> **Note:** If a match doesn't appear in the dashboard's Matches list or you see `ERROR: Frames directory not found`, make sure the match has been processed by the pipeline first.

### 2.3 Creating a Labeling Project

1. Open Label Studio at `http://<server-address>:8080`
2. Click **Create** (top-right corner)
3. Enter a **Project Name** -- use the match name for clarity (e.g., `LastGame-Test`)
4. Optionally add a description (e.g., "Ball detection labels for LastGame-Test")
5. Click **Save** (don't import data yet -- configure the interface first)

### 2.4 Configuring the Labeling Interface

Before importing tasks, set up the correct annotation template:

1. Inside your project, click **Settings** (gear icon, top-right)
2. Go to the **Labeling Interface** tab
3. Click **Browse Templates** and search for **Object Detection with Bounding Boxes**
4. Select the template
5. In the label configuration, **remove** any default labels and add a single label: **`ball`**
6. The XML config should look like:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="ball" background="red"/>
  </RectangleLabels>
</View>
```

1. Click **Save**

> **Warning:** The label name must be exactly `ball` (lowercase). The training pipeline expects this specific label.

### 2.5 Importing Tasks Into the Project

1. In your project, click **Import** (or go to Settings -> Cloud Storage)
2. Click **Upload Files**
3. Navigate to `/tank/labeling/<match_name>/labelstudio/tasks.json` on the server
4. Upload the `tasks.json` file
5. Label Studio will show the number of tasks imported

> **Tip:** If you prefer, you can also use Label Studio's **Local Storage** sync feature:
>
> 1. First, run `bash scripts/labelstudio_import.sh <match_name>` to generate the `tasks.json` file
> 1. In Label Studio, go to Settings -> Cloud Storage -> Add Source Storage
> 1. Choose **Local files**
> 1. Set the absolute path to `/label-studio/data/labeling/<match_name>/labelstudio/`
> 1. Click **Test Connection** to verify, then **Add Storage**
> 1. Click **Sync Storage** to import tasks

After import, the Data Manager shows all frames with their status (unlabeled/labeled).

---

## Part 3: Labeling Ball Positions

### 3.1 Opening a Task

From the Data Manager, click any row to open the labeling view. You'll see:

- The hard frame image (full 360 equirectangular view)
- The label toolbar on the left with the **ball** label
- Navigation buttons to move between tasks

### 3.2 Drawing a Bounding Box

1. Click the **ball** label in the left panel (or press the keyboard shortcut)
2. Click and drag on the image to draw a rectangle around the ball
3. Make the box tight -- just big enough to contain the ball
4. Click **Submit** to save and move to the next task

> **Tip:** The ball is typically small (10-30 pixels wide in the detection-resolution image). Zoom in using your scroll wheel or the zoom controls to get an accurate box.

### 3.3 Using Pre-Annotations

If the import script found predicted bounding boxes for a frame, you'll see a pre-drawn box with a dashed border. This is the model's best guess.

- **If the box is correct:** Simply click **Submit** -- the pre-annotation is accepted as your label
- **If the box is close but off:** Drag the corners or edges to adjust it, then click **Submit**
- **If the box is wrong:** Delete it (select it and press Delete/Backspace), then draw a new one
- **If there's no ball visible:** Skip the frame or submit with no annotation

### 3.4 Handling Difficult Frames

You'll encounter frames where labeling isn't straightforward:

- **Ball is partially occluded (behind a player)** -- Draw the box where the ball is, even if partially hidden. The model needs to learn these cases.
- **Ball is blurry or motion-smeared** -- Draw the box around the center of the blur streak. An approximate box is better than no box.
- **No ball visible in the frame** -- This happens with lost ball run triggers. Submit with no annotation -- this teaches the model that there is nothing to detect here.
- **Multiple balls visible (e.g., spare balls on the sideline)** -- Only label the **game ball** (the one in play on the field). Ignore sideline/spare balls.
- **Ball is at the very edge of the frame** -- Label it even if partially cropped. Include as much of the visible ball as possible.

### 3.5 Keyboard Shortcuts

Label Studio has built-in shortcuts to speed up labeling:

| Shortcut | Action |
| --- | --- |
| `1` | Select the first label (ball) |
| `Ctrl+Enter` | Submit and go to next task |
| `Ctrl+Backspace` | Delete selected annotation |
| `Ctrl+Z` | Undo last action |
| `+` / `-` | Zoom in/out |
| `Ctrl+[` / `Ctrl+]` | Previous/next task |

### 3.6 Tips for Accurate Labels

- **Be consistent** -- draw boxes the same way for every frame
- **Tight boxes** -- the box should touch all edges of the ball
- **Center matters more than size** -- if you're unsure about the exact edges, focus on centering the box on the ball
- **Don't over-think it** -- spending 2-3 seconds per frame is fine. The model benefits more from quantity than perfection
- **5-10 minutes per session** -- even a quick session after each match makes a difference

---

## Part 4: Exporting Labels

### 4.1 Exporting in YOLO Format

After labeling all (or some) frames in a project:

1. Open the project in Label Studio
2. Click **Export** (top-right)
3. Select **YOLO** as the export format
4. Click **Export** to download a ZIP file

The ZIP contains a `labels/` directory with one `.txt` file per labeled image.

### 4.2 Uploading Labels via the Dashboard (Recommended)

The simplest way to get labels back to the server is the **Upload** button in the dashboard:

1. Open the Soccer360 Dashboard at `http://<server-address>:8088`
2. Switch to the **Labeling & Training** workspace tab
3. Find your match in the list
4. Click the **Upload** button next to the match name
5. Select the YOLO export ZIP file you downloaded from Label Studio
6. A toast notification confirms the number of label files extracted (e.g., "Uploaded 47 label files for Match-A.")
7. The label count updates automatically in the match row

The dashboard extracts the `.txt` label files from the ZIP and places them in `/tank/labeling/<match_name>/labels/` automatically. No manual file handling needed.

> **Tip:** This is the recommended method because it handles file naming and placement automatically and gives immediate feedback on how many labels were extracted.

### 4.3 Placing Labels Manually (Alternative)

If you prefer to handle files manually (e.g., via SSH), extract the YOLO label files to the correct location:

```bash
# Extract the ZIP (replace with your actual downloaded filename)
unzip export-*.zip -d /tmp/ls-export/

# Copy label files to the match's labels directory
mkdir -p /tank/labeling/<match_name>/labels/
cp /tmp/ls-export/labels/*.txt /tank/labeling/<match_name>/labels/

# Clean up
rm -rf /tmp/ls-export/
```

> **Important:** The label `.txt` files must have the same base name as their corresponding frame images. For example:
>
> - Image: `/tank/labeling/LastGame-Test/frames/frame_000014.jpg`
> - Label: `/tank/labeling/LastGame-Test/labels/frame_000014.txt`

**Verify the export:**

```bash
# Check that labels exist
ls /tank/labeling/<match_name>/labels/ | head -5

# Check a label file (should have: class_id x_center y_center width height)
cat /tank/labeling/<match_name>/labels/frame_000014.txt
```

A typical YOLO label line looks like:

```text
0 0.523 0.341 0.015 0.028
```

This means: class 0 (ball), centered at 52.3% x / 34.1% y, with width 1.5% and height 2.8% of the image.

> **Note:** Although the pipeline detects both players (COCO class 0) and the ball (COCO class 32) during processing, the training dataset uses a single-class format where class 0 = ball. This is a YOLO training convention for single-class datasets and is handled automatically by the dataset builder.

---

## Part 5: Building the Dataset and Training

Once you have labeled frames from one or more matches, you can build a training dataset and fine-tune the model.

### 5.1 Using the Dashboard (Recommended)

The Soccer360 Dashboard at `http://<server-address>:8088` has a built-in training interface:

1. Open the dashboard and switch to the **Labeling & Training** workspace tab
2. Review the labeling status — it shows frame counts, imported task counts, and label counts per match
3. If you haven't uploaded labels yet, click **Upload** next to the match and select your YOLO export ZIP; a toast confirms the label count
4. Click **Build Dataset** — a confirmation modal appears before the build starts; a toast confirms completion
5. Once the build completes, click **Train Model** (adjust epochs if desired, default is 50) — a modal shows training parameters for confirmation
6. Training progress is shown in the log area below the buttons

> **Note:** The dashboard now builds the dataset with native Python logic and starts training with `python -m src.cli train`. Training uses the GPU and can take 30 minutes to 2 hours depending on dataset size and number of epochs. The pipeline continues to work while training runs.

### 5.2 Using the Command Line

Alternatively, run the build helper and CLI directly on the server:

#### Step 1: Build the dataset

```bash
bash scripts/build_dataset.sh
```

This scans all matches under `/tank/labeling/`, collects paired image/label files, and creates an 80/20 train/val split at `/tank/labeling/dataset/`.

Output:

```text
================================================
Soccer360 Dataset Builder
  Scanning: /tank/labeling
  Output:   /tank/labeling/dataset
  Val ratio: 0.2
================================================
Found 47 labeled images across 3 matches:
  Match-A: 22 images
  Match-B: 15 images
  Match-C: 10 images

Dataset built: 38 train, 9 val
YAML: /tank/labeling/dataset/dataset.yaml
```

#### Step 2: Train the model

```bash
soccer360 train --epochs 50 --data /tank/labeling/dataset/dataset.yaml
```

This fine-tunes the YOLO model for 50 epochs using your labeled data. When training completes:

- The best model is saved to `/tank/models/ball_best.pt`
- A versioned copy is kept at `/tank/models/ball_model_YYYYMMDD_HHMM/`
- Future ingest jobs use it if the dashboard ingest selector is set to `Auto` or pinned to `ball_best.pt`

`bash scripts/train_ball.sh 50` remains available as a helper wrapper around the same training flow.

### 5.3 Verifying the New Model

After training, set the dashboard ingest selector to `Auto` or pin `ball_best.pt` if you want the next video processed to use the improved model. To verify:

1. **Check the dashboard** -- the Available Models list under Active Learning shows all `.pt` files with the active model marked
2. **Check the logs** -- look for the model resolution line:

```text
Model resolved: /tank/models/ball_best.pt (source=runtime.auto)
```

1. **Reprocess a previous match** to compare results:
   Open the match page from the dashboard, click **Remove Match Family**, confirm in the modal, then use the **Staging** panel to move the restored `*_reprocess` source file back into ingest.

---

## Part 6: The Weekly Improvement Cycle

For the best results, follow this weekly rhythm:

| Day | Activity | Time |
| --- | --- | --- |
| **Match days** | Process recordings (automatic) | ~90 min per match |
| **After processing** | Import hard frames to Label Studio | 2 min per match |
| **During the week** | Label hard frames in Label Studio | 5-10 min per session |
| **Before next match day** | Build dataset + train model | 30-120 min (automated) |

**Practical workflow:**

1. **Process games** -- drop videos into `/tank/ingest/`, the worker handles the rest
2. **Import** -- click **Import** in the dashboard (or run `bash scripts/labelstudio_import.sh <match>`)
3. **Label** -- open Label Studio, label frames whenever you have a few minutes
4. **Upload** -- click **Upload** in the dashboard and select the YOLO export ZIP (or extract manually)
5. **Train** -- use the dashboard's Build Dataset + Train buttons (or run `bash scripts/build_dataset.sh` then `soccer360 train --epochs 50 --data /tank/labeling/dataset/dataset.yaml`)
6. **Next games are better** -- set the dashboard ingest selector to `Auto` or pin `ball_best.pt`, then reprocess or ingest the next game

> **Tip:** You don't need to label every hard frame. Even labeling 20-30 frames per match yields meaningful improvement. Focus on frames where you can clearly see the ball.

---

## Reference

### Directory Structure

```text
/tank/labeling/
  <match_name>/
    frames/                      Auto-exported hard frame images
      frame_000014.jpg
      frame_000287.jpg
      ...
    hard_frames.json             Manifest with trigger info + exported bbox metadata
    labels/                      YOLO labels (from Label Studio export)
      frame_000014.txt
      frame_000287.txt
      ...
    labelstudio/                 Label Studio import files
      tasks.json
  dataset/                       Built training dataset
    train/
      images/
      labels/
    val/
      images/
      labels/
    dataset.yaml
```

### Hard Frame Trigger Types

| Trigger | Config Key | Default | Description |
| --- | --- | --- | --- |
| Low confidence | `active_learning.low_conf_min` / `low_conf_max` | 0.10 -- 0.50 | Detection confidence in the uncertain range |
| Lost ball run | `active_learning.lost_run_frames` | 5 | Consecutive frames with no detection |
| Jump rejection | `active_learning.jump_trigger_px` | 150 | Ball position jumped unrealistically far |

Additional gating:

| Config Key | Default | Description |
| --- | --- | --- |
| `active_learning.export_every_n_frames` | 2 | Sample 1 in N low-confidence frames |
| `active_learning.export_max_frames` | 600 | Maximum exported frames per match |

### Label Format (YOLO)

Each `.txt` label file contains one line per object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to 0.0 -- 1.0 relative to the image dimensions. For Soccer360, class 0 is always `ball`.

Example: `0 0.523 0.341 0.015 0.028`

### Troubleshooting

#### Label Studio won't load images

The images are served from the container's local filesystem. Verify the volume mount:

```bash
docker compose exec labelstudio ls /label-studio/data/labeling/
```

You should see your match directories. If not, check that `/tank/labeling` is correctly mounted in `docker-compose.yml`.

#### "No tasks" after import

- Make sure you used the correct `tasks.json` from `/tank/labeling/<match>/labelstudio/`
- Check that the labeling interface is configured (Settings -> Labeling Interface)
- Try re-running `bash scripts/labelstudio_import.sh <match>`

#### Labels don't match images after export

YOLO export file names must match the frame file names exactly. If Label Studio renames them during export:

```bash
# Check what Label Studio exported
ls /tmp/ls-export/labels/

# If names are different, rename to match frame names
cd /tank/labeling/<match>/labels/
for f in *.txt; do
    # Adjust renaming logic based on actual exported names
    echo "$f"
done
```

#### Build dataset shows "No image/label pairs found"

This means there are no matches where both `frames/*.jpg` and `labels/*.txt` exist with matching names. Check:

```bash
# See what's available
for d in /tank/labeling/*/; do
    echo "=== $(basename $d) ==="
    echo "  Frames: $(ls $d/frames/*.jpg 2>/dev/null | wc -l)"
    echo "  Labels: $(ls $d/labels/*.txt 2>/dev/null | wc -l)"
done
```

#### Training fails with "Dataset not built"

Run `bash scripts/build_dataset.sh` first (or click Build Dataset in the dashboard). Training needs `/tank/labeling/dataset/dataset.yaml` to exist before you run `soccer360 train --epochs 50 --data /tank/labeling/dataset/dataset.yaml`.

#### Model doesn't seem better after training

- Check that `ball_best.pt` was updated: `ls -la /tank/models/ball_best.pt`
- Check the dashboard ingest selector or `/api/inference/model` to confirm future jobs are pointed at that model
- Make sure you labeled enough frames (20+ is a good minimum across all matches)
- Check training log for validation metrics -- mAP should be improving across epochs
- The model needs diverse examples: label frames from different matches, lighting conditions, and ball positions
