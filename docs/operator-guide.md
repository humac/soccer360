# Soccer360 -- Operator Guide

A guide for day-to-day operation of the Soccer360 pipeline: ingesting match recordings, monitoring processing, retrieving outputs, and improving detection through labeling.

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [What You Need](#what-you-need)
  - [How the System Works](#how-the-system-works)
- [Ingesting a Match Recording](#ingesting-a-match-recording)
  - [Preparing Your Video](#preparing-your-video)
  - [Dropping a File for Processing](#dropping-a-file-for-processing)
  - [Safe Copy Method](#safe-copy-method)
- [Monitoring Processing](#monitoring-processing)
  - [Checking the Worker Status](#checking-the-worker-status)
  - [Viewing Logs](#viewing-logs)
  - [Understanding Processing Phases](#understanding-processing-phases)
- [Retrieving Outputs](#retrieving-outputs)
  - [Broadcast Video](#broadcast-video)
  - [Tactical Wide View](#tactical-wide-view)
  - [Highlight Clips](#highlight-clips)
  - [Metadata and Diagnostics](#metadata-and-diagnostics)
- [Understanding Output Modes](#understanding-output-modes)
- [Active Learning: Improving the Model](#active-learning-improving-the-model)
  - [What Are Hard Frames?](#what-are-hard-frames)
  - [Labeling in Label Studio](#labeling-in-label-studio)
  - [The Weekly Improvement Cycle](#the-weekly-improvement-cycle)
- [Reprocessing a Match](#reprocessing-a-match)
- [Troubleshooting](#troubleshooting)
  - [Video Not Being Picked Up](#video-not-being-picked-up)
  - [Processing Seems Stuck](#processing-seems-stuck)
  - [Output Quality Issues](#output-quality-issues)
  - [No Highlights Generated](#no-highlights-generated)
  - [Getting Help](#getting-help)

---

## Overview

Soccer360 takes a 360-degree match recording (from a camera like the Insta360 X5) and automatically produces:

- **Broadcast video** -- an auto-follow view that tracks the ball, mimicking a traditional TV broadcast
- **Tactical wide view** -- a fixed wide-angle perspective of the full pitch
- **Highlight clips** -- short clips of goals, shots, and other key moments

You don't need to edit anything manually. Drop the video in, wait for processing, and collect the results.

## Getting Started

### What You Need

- A 360-degree match recording in **equirectangular MP4 format** (typically 5760x2880 or similar)
- Network access to the server's `/tank/ingest/` folder (or SSH/SCP access)
- Access to the server for checking logs (optional but recommended)

> **Note:** If you recorded with an Insta360 camera, you must first stitch and export to equirectangular MP4 using Insta360 Studio before ingesting. Raw `.insv` files cannot be processed directly.

### How the System Works

The pipeline runs as an always-on background service (the "watcher"). When you place a video file in the ingest folder, the watcher:

1. Detects the new file
2. Waits for it to finish copying (50 seconds of stable file size)
3. Processes it through detection, tracking, camera generation, and rendering
4. Saves outputs to the processed folder
5. Archives the original recording (if configured)

A typical 1-hour match at 5.7K resolution takes approximately 60-90 minutes to process.

## Ingesting a Match Recording

### Preparing Your Video

Before ingesting, ensure your video meets these requirements:

| Requirement | Details |
|-------------|---------|
| Format | MP4 or MOV |
| Projection | Equirectangular (full 360x180) |
| Resolution | 5760x2880 recommended (other equirectangular resolutions work) |
| Duration | Full match length (no need to trim) |

> **Warning:** Do not ingest partial recordings, corrupted files, or non-equirectangular video. The pipeline will fail or produce unusable output.

### Dropping a File for Processing

The simplest method -- copy the file directly:

```bash
cp match_2024_01_15.mp4 /tank/ingest/
```

The watcher picks it up automatically after the file size stabilizes.

### Safe Copy Method

For large files over a network, use the atomic copy method to prevent the watcher from starting on a partially copied file:

```bash
# Step 1: Copy with a .part extension (watcher ignores .part files)
cp match_2024_01_15.mp4 /tank/ingest/match_2024_01_15.mp4.part

# Step 2: Rename to remove the .part extension (triggers processing)
mv /tank/ingest/match_2024_01_15.mp4.part /tank/ingest/match_2024_01_15.mp4
```

> **Tip:** The watcher also ignores files ending in `.tmp` and `.uploading`. Use any of these extensions during transfer.

### Multiple Matches

You can drop multiple files at once. They are processed sequentially in the order detected:

```bash
cp match1.mp4 match2.mp4 match3.mp4 /tank/ingest/
```

## Monitoring Processing

### Checking the Worker Status

See if the worker service is running and healthy:

```bash
docker compose ps worker
```

You should see the worker with status `Up` and `(healthy)`. The health check runs every 60 seconds and verifies that storage mounts and GPU are accessible. If the status shows `(unhealthy)`, contact your administrator -- the worker may have lost access to `/tank`, `/scratch`, or the GPU.

Similarly, check Label Studio:

```bash
docker compose ps labelstudio
```

Label Studio also has a health check and should show `(healthy)` when ready to accept connections.

### Viewing Logs

Follow the live processing log:

```bash
docker compose logs -f worker
```

Key log messages to watch for:

| Log Message | Meaning |
|-------------|---------|
| `Processing: <filename>` | Pipeline has started on a new file |
| `Model resolved: <path> (source=<source>)` | Which detection model is being used |
| `Phase 'detection' completed in 1234.567s` | Ball detection finished with elapsed time |
| `Phase 'broadcast_reframe' completed in 567.890s` | Broadcast video rendered with elapsed time |
| `PIPELINE COMPLETE: <filename> (X.Y min)` | All outputs ready |
| `mode: no_detect` | No detection model available; reduced output |

> **Tip:** Each processing phase now logs its wall-clock duration. This makes it easy to identify which phase is taking longest and whether processing times are changing over time.

Press `Ctrl+C` to stop following logs (the worker keeps running).

### Understanding Processing Phases

| Phase | What Happens | Duration (typical) |
|-------|-------------|-------------------|
| 1. Detection | YOLO finds the ball in each frame using the GPU | 15-25 min |
| 2. Tracking | Ball positions are stabilized and smoothed | < 1 min |
| 2.5. Hard frames | Difficult frames exported for future labeling | < 1 min |
| 3. Camera path | Virtual camera angles calculated per frame | < 1 min |
| 4. Broadcast | Auto-follow video rendered (12 parallel workers) | 20-40 min |
| 5. Tactical | Wide-angle view rendered in parallel | 10-20 min |
| 6. Highlights | Key moments detected and clips exported | 2-5 min |
| 7. Export | Outputs organized, metadata written | < 1 min |
| 8. Cleanup | Temporary scratch files removed | < 1 min |

## Retrieving Outputs

After processing completes, find outputs at:

```
/tank/processed/<match_name>/
```

The `<match_name>` is derived from the input filename (without extension).

### Broadcast Video

**File:** `broadcast.mp4`

The main output. An auto-follow perspective view that tracks the ball across the pitch, similar to a traditional TV broadcast. Resolution: 1920x1080, H.264 encoded.

### Tactical Wide View

**File:** `tactical_wide.mp4`

A fixed wide-angle view (120-degree FOV) showing most of the pitch. Useful for tactical analysis. Same resolution and codec as broadcast.

### Highlight Clips

**Location:** `/tank/highlights/<match_name>/`

Short clips (typically 8-15 seconds each) capturing key moments:
- Fast ball movement (shots, long passes)
- Sharp direction changes (deflections, saves)
- Goal-box activity (shots on goal, corner kicks)

Each clip is a standalone MP4 file.

> **Note:** Highlights are only generated in normal processing mode. If the pipeline ran in NO_DETECT mode (no ball detection model available), no highlights are produced.

### Metadata and Diagnostics

**File:** `metadata.json`

Contains processing details for troubleshooting:
- Processing mode (`v1_bootstrap`, `legacy`, or `no_detect`)
- Model path and source used
- Per-phase timing breakdown (how long each phase took)
- Detection and tracking quality stats (detection count, frames with ball found)
- GPU utilization snapshot captured after detection
- Ingest archival status

The `phase_metrics` section in `metadata.json` is especially useful for understanding performance:

```json
"phase_metrics": {
  "phase_timings_sec": {
    "detection": 1234.5,
    "tracking": 2.3,
    "hard_frames": 1.1,
    "camera": 0.8,
    "broadcast_reframe": 567.9,
    "tactical_reframe": 321.4,
    "highlights": 15.2,
    "export": 3.1
  },
  "stats": {
    "detection_count": 48000,
    "frames_processed": 54000,
    "track_frames_total": 54000,
    "track_frames_with_ball": 41000,
    "gpu_snapshot_post_detection": {
      "gpu_utilization_pct": 45,
      "memory_used_mb": 8192,
      "memory_total_mb": 24576,
      "temperature_c": 72
    }
  }
}
```

> **Tip:** Compare `track_frames_with_ball` against `track_frames_total` to get a sense of how well the model is detecting the ball. A low ratio suggests the model needs improvement via the active learning workflow.

**Other diagnostic files:**
- `detections.jsonl` -- raw ball detections per frame
- `tracks.json` -- stabilized ball positions
- `camera_path.json` -- virtual camera angles per frame
- `foi_meta.json` -- Field-of-Interest filter metadata
- `hard_frames.json` -- manifest of hard frames exported for labeling

## Understanding Output Modes

The pipeline has three processing modes depending on model availability:

| Mode | Ball Tracking | Broadcast | Tactical | Highlights | When |
|------|:---:|:---:|:---:|:---:|------|
| **V1 Bootstrap** | Yes | Auto-follow | Yes | Yes | Detection model available (default) |
| **Legacy** | Yes | Auto-follow | Yes | Yes | Config uses legacy tracker |
| **NO_DETECT** | No | Static framing | Yes | No | No model available |

Check `metadata.json` for the `"mode"` field to see which mode was used.

> **Tip:** If you see `"mode": "no_detect"` in metadata, the broadcast video will have a fixed center framing instead of tracking the ball. Contact your administrator to ensure a detection model is available.

## Active Learning: Improving the Model

The pipeline automatically improves over time through a labeling feedback loop.

### What Are Hard Frames?

During each processing run, the pipeline identifies frames where the ball detection model struggled:

- **Low confidence detections** -- the model found something but wasn't sure it was a ball
- **Lost ball streaks** -- consecutive frames where no ball was detected at all
- **Position jumps** -- the detected ball position jumped unrealistically between frames

These "hard frames" are exported as JPEG images to `/tank/labeling/<match_name>/frames/` for human review.

### Labeling in Label Studio

Label Studio runs alongside the pipeline for annotating hard frames:

1. **Open Label Studio** at `http://<server-address>:8080`
2. **Import hard frames** (your administrator will run the import script)
3. **Create a bounding box** around the ball in each image
   - If no ball is visible, skip the frame
   - Draw a tight box around the ball only
4. **Export annotations** in YOLO format when done

> **Tip:** Even 5-10 minutes of labeling per week makes a meaningful difference. Focus on frames where you can clearly see a ball that the model missed.

### The Weekly Improvement Cycle

1. **Games are processed** -- hard frames are auto-exported
2. **You label hard frames** -- 5-10 minutes in Label Studio
3. **Administrator builds dataset and trains** -- one command each
4. **Next games are better** -- the worker automatically uses the improved model

## Reprocessing a Match

To reprocess a match with an updated model or different settings:

```bash
docker compose run --rm worker soccer360 process /tank/ingest/match.mp4
```

Or, if the original was archived:

```bash
docker compose run --rm worker soccer360 process /tank/archive_raw/match_<job_id>.mp4
```

> **Note:** If the watcher has already processed this file, you may need to ask your administrator to reset the dedupe state first. See the Admin Guide for details.

To keep intermediate scratch files for debugging:

```bash
docker compose run --rm worker soccer360 process /path/to/match.mp4 --no-cleanup
```

## Troubleshooting

### Video Not Being Picked Up

**Symptoms:** File is in `/tank/ingest/` but nothing happens.

**Possible causes:**
- **File still copying.** The watcher waits 50 seconds of stable file size. Wait and check logs.
- **Wrong extension.** Only `.mp4`, `.mov`, and `.insv` are accepted.
- **Hidden file.** Files starting with `.` (dot) are ignored.
- **Staging suffix.** Files ending in `.part`, `.tmp`, or `.uploading` are ignored (remove the suffix).
- **Already processed.** The watcher remembers files it has processed. Check with your administrator about resetting dedupe state.
- **Worker not running.** Check with `docker compose ps worker`.

### Processing Seems Stuck

**Symptoms:** Logs show the pipeline started but no progress for a long time.

**Possible causes:**
- **Phase 1 (Detection) is the longest phase.** A 1-hour match can take 15-25 minutes for detection alone. Check logs for frame progress.
- **GPU memory issue.** If the GPU runs out of memory, detection may fail silently. Ask your administrator to check GPU status.
- **Corrupt video file.** FFmpeg may hang on corrupt input. Try `ffprobe /tank/ingest/<file>` to verify the file is readable.

### Output Quality Issues

| Issue | Likely Cause | Remedy |
|-------|-------------|--------|
| Camera not following the ball | Detection model missing balls frequently | Label hard frames to improve model |
| Camera jittery / oscillating | Ball detections are noisy | Check FoI settings; may need wider yaw window |
| Broadcast shows wrong field | FoI center pointing at adjacent field | Ask admin to adjust `field_of_interest.center_yaw_deg` |
| Black frames or artifacts | Source video has stitching issues | Re-export from Insta360 Studio |
| Highlight clips are irrelevant | Heuristic thresholds need tuning | Ask admin to adjust highlight parameters |

### No Highlights Generated

- Check if the pipeline ran in NO_DETECT mode (`metadata.json` > `"mode"`)
- In NO_DETECT mode, highlights are disabled because there's no ball track to analyze
- Ensure a detection model is available and the pipeline runs in normal mode

### Getting Help

- Check the processing logs: `docker compose logs -f worker`
- Review `metadata.json` in the output folder for processing details
- Contact your system administrator for configuration changes, model issues, or server problems
