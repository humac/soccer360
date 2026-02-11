# Local 360 Soccer Auto-Tracking & Processing Server

## Overview
This server functions as a fully automated local sports video processing system designed for **zero-touch processing**.

## Workflow
1. Record a 360 soccer match (Insta360 X5/X4, etc.)
2. Drop the raw file into the ingest folder
3. Server automatically:
   - Detects/tracks the ball
   - Generates an auto-follow broadcast video
   - Creates a tactical view
   - Exports highlights
4. Review outputs later (**no manual editing**)

---

## Hardware Configuration

### Server
- Dell PowerEdge (dual Xeon)
- 256 GB RAM
- NVIDIA Tesla P40 (24GB VRAM)
- Nothing else running on the system
- Dedicated to video/AI processing

### Storage
**NVMe (fast scratch)**
- 512GB NVMe
- Used **only** for active processing
- Mounted: `/scratch`

**SSD storage**
- 4TB SSD
- Used for ingest + final outputs + models
- Mounted: `/tank`

---

## Storage Layout

```text
/scratch                    ← active processing only (auto cleaned)
/tank
  ├── ingest/               ← drop new raw videos here
  ├── processing/           ← temp working (optional)
  ├── processed/            ← final broadcast videos
  ├── highlights/           ← highlight clips
  ├── archive_raw/          ← optional raw storage
  ├── models/               ← trained ball tracking models
  └── labeling/             ← training images/labels
```

### Storage Rules
- Always process from `/scratch`
- Never process directly from `/tank`
- Final outputs saved to `/tank/processed`
- Raw files auto-delete or move after processing

---

## OS & Base Software

### OS
- Ubuntu 22.04 LTS (bare metal)  
  *(no hypervisor, no VMs required)*

### Why
- Simplest NVIDIA + CUDA setup
- No GPU passthrough headaches
- Maximum performance
- Easier maintenance

---

## Core Software Stack

### GPU / CUDA
Install:
- NVIDIA datacenter driver (535+)
- CUDA toolkit
- `nvidia-container-toolkit` (for Docker GPU)

Verify:
- `nvidia-smi`  
  *(should show Tesla P40)*

### Docker Environment
Everything runs in Docker.

Containers:
- Soccer pipeline worker
- Watcher/automation service
- Labeling tool (Label Studio optional)
- Model training container
- Optional web viewer

Install:
- `docker`
- `docker-compose`
- `nvidia-container-toolkit`

---

## Processing Pipeline Design

### Step 1 — Ingest
Watch folder:
- `/tank/ingest`

When a new file appears:
- Copy → `/scratch`
- Start pipeline automatically

### Step 2 — Ball Detection
Use:
- YOLOv8 custom ball model
- GPU inference (P40)

Outputs:
- Ball coordinates per frame
- Confidence score
- Lost tracking flag

### Step 3 — Tracking + Camera Path
Convert ball position → yaw/pitch

Smooth movement:
- Jitter reduction
- Max pan speed
- Zoom logic
- Fallback if ball lost

Generate:
- `camera_path.json`

### Step 4 — 360 Reframing
Using:
- `py360convert` (or equivalent)

Generate:
- `broadcast_follow.mp4`
- `tactical_wide.mp4`

Resolution target:
- 1080p60 (default)
- Configurable

### Step 5 — Highlights
Heuristic detection:
- Shots on goal
- Rapid motion clusters
- Box entries

Export:
- `/tank/highlights/*.mp4`

---

## Auto Cleanup Rules

### Scratch cleanup
Delete from `/scratch`:
- After successful processing, **or**
- If >48h old

### Raw retention
Keep raw:
- Optional 7–14 days, **or**
- Delete immediately after success

---

## Model Improvement Workflow
System must support continuous improvement.

### Export hard frames
Automatically capture:
- Lost ball frames
- Low confidence frames
- Cluster confusion

Save:
- `/tank/labeling/<game>/`

### Labeling
Use:
- Label Studio (Docker)

### Retraining
Fine-tune YOLO model locally:
- `/tank/models/ball_model_vX.pt`

Models are versioned and reusable.

---

## Expected Performance
Per 1 hour match:

| Task | Time |
|---|---|
| Detection/tracking | 20–40 min |
| Reframe/export | 30–50 min |
| Highlights | 10–20 min |
| **Total pipeline** | **~1–1.5 hr** |

Runs fully unattended.

---

## Future Expansion
- Player tracking
- Auto player clips
- Web dashboard
- Plex library integration
- Reprocess old matches with new models

---

## Philosophy
This server acts as a local Veo-style system without subscription.

All processing is:
- Local
- Private
- Improvable over time
