# Soccer360 Detection Improvement Plan

## Problem Statement

Ball detection is critically underperforming:

| Metric | 1st Half | 2nd Half |
|---|---|---|
| Total frames | 50,303 | 48,227 |
| Ball detections | ~1,700 | 823 (1.7%) |
| Tracked ball frames | 1,194 (2.4%) | 505 (1.0%) |
| Lost frames | ~97.6% | 98.3% |
| Ball confidence | — | median 0.167 |

The broadcast camera is essentially static — following player cluster drift only because there's no ball to track.

### Root Causes

1. **Tiny training dataset** — 80 train + 20 val images. Model confidence is barely above the 0.1 threshold.
2. **Source exported at 4K (3840x1920), not 5.7K/8K** — half the available angular resolution lost at export.
3. **~55% of pixels wasted** — camera at sideline, only ~160° of the 360° view covers the field. Detection runs on the full equirectangular including the useless back side.
4. **Equirectangular distortion** — YOLO trained on perspective images, not equirect.

---

## Phase 0: Quick Wins (immediate, no code changes)

### 0.1 Re-export source at 5.7K (5760x2880)
- Insta360 Studio export settings:
  - Resolution: **5760 x 2880**
  - Bitrate: **100-120 Mbps** (50 Mbps causes compression blur on the ball)
  - Frame rate: 29.97 (unchanged)
  - Encoding: H.265 (unchanged)
  - Export as: 360 Video (unchanged)
- Result: ~2.25x more pixels, ball is larger and sharper in source

### 0.2 Bump detection resolution
- `configs/pipeline.yaml` change: `detection.img_size: 1920` (already done)
- At 5.7K source, YOLO resizes internally to 1920x960
- Field goes from ~427px to ~854px wide at detection scale
- Ball goes from ~13px to ~26px median — much more reliably detectable

### 0.3 Downscale source for reframing (optional, for speed)
- `reframer.source_downscale: [3840, 1920]` — renders from downscaled frames
- Broadcast/tactical output is 1920x1080 anyway, no visible quality loss
- Saves ~50% reframe time (avoids processing 5.7K frames for 1080p output)

### 0.4 Reprocess and evaluate
- Drop 5.7K export into `/tank/ingest`
- Compare tracking rate against the 1-2.4% baseline
- Target: >10% ball tracking rate from resolution improvement alone

### 0.5 Label hard frames and re-train
- Pipeline exports up to 600 hard frames per run to `/tank/labeling/`
- Label ball bounding boxes in Label Studio
- Build dataset via dashboard
- Re-train model — going from 100 to 700+ labeled frames should significantly improve confidence
- Reprocess again with new model

### 0.6 Iterate
- Each processed game generates more hard frames
- Label → re-train → reprocess cycle compounds quality
- Target: >30% ball tracking rate after 2-3 training iterations

---

## Phase 1: Multi-View Detection Experiment

> Only proceed after Phase 0 improvements are measured.

**Goal:** Test whether rendering 3 perspective crops from the 360 source and detecting on each improves ball detection vs the single equirectangular approach.

**Why this helps:** Instead of wasting detection resolution on the useless back side of the camera, focus 3 views on the field only. Each 75° view at 960px gives ~12.8 px/deg vs ~5.3 px/deg for full equirect at img_size=1920.

**Deliverable:** Standalone `scripts/multiview_experiment.py` that:
1. Renders 3 perspective crops (left/center/right) from a processed match video
2. Runs YOLO on each crop
3. Back-projects detections to equirectangular space
4. Fuses detections (angular NMS)
5. Compares against baseline `detections.jsonl`
6. Outputs a comparison report

**Key technical work:**
- `perspective_to_equirect()` — inverse projection from perspective pixel to equirect pixel
- 3-view layout: `center_yaw - offset`, `center_yaw`, `center_yaw + offset`
- Detection fusion: greedy angular NMS across views

**Decision gate:** If multi-view shows >10% detection improvement over Phase 0 results, proceed to Phase 2. Otherwise, focus on training data.

---

## Phase 2: Pipeline Integration (if Phase 1 validates)

**Goal:** Integrate multi-view detection into the pipeline, gated behind `multiview.enabled: false`.

**New module `src/multiview.py`:**
- `ViewPlanner` — computes 3 view parameters from FoI config
- `ViewRenderer` — renders perspective numpy arrays per frame
- `DetectionFuser` — back-projects + fuses detections
- `perspective_to_equirect()` — core projection math

**Pipeline change:** Single insertion point in `src/pipeline.py` at the detection phase — conditional dispatch to `run_streaming_multiview()`. Everything downstream unchanged — same `detections.jsonl` format.

**Config:**
```yaml
multiview:
  enabled: false
  view_fov: 75
  view_pitch: -5.0
  field_width_deg: null    # auto from field_of_interest.yaw_window_deg
  detection_resolution: [960, 540]
  fusion:
    proximity_threshold_deg: 3.0
    min_confidence: 0.15
```

---

## Phase 3: Dual Flat Camera Support (hardware upgrade path)

> Pursue if/when upgrading to DJI Action 5 Pro or GoPro Hero 13 dual-camera rig.

**Why:** Two flat 4K cameras at 100° FOV each give ~38 px/deg — a 14x improvement over current equirect detection, matching Veo Cam 3's hardware approach.

**Architecture:** No pipeline rewrite needed. The `src/multiview.py` module from Phase 2 handles both paths — the back-projection and fusion math is identical whether perspective frames come from virtual crops (360 camera) or real cameras (flat cameras).

**Additional work needed:**
- Camera calibration (one-time): intrinsics + relative pose → stored as JSON
- Stitcher: project two flat feeds into equirectangular for broadcast/tactical rendering
- Frame sync: timecode or audio alignment between cameras
- Pipeline dispatch: config flag for single-360 vs dual-flat input mode

**What stays untouched:** tracker, player cluster, camera path, highlights, exporter, dashboard — all consume the same `detections.jsonl` format regardless of input source.

---

## Phase 4: Manual Calibration and Dashboard (v2)

> Only if auto-centering proves insufficient for view placement.

- Dashboard UI for drawing field polygon on extracted frame
- Calibration metadata format (JSON)
- CLI support: `--calibration path/to/calibration.json`
- ViewPlanner uses calibration when present, falls back to auto-centering

---

## Hardware Comparison Reference

| Setup | Effective px/deg on field | YOLO distortion | Ops complexity | Cost |
|---|---|---|---|---|
| Insta360 X5 4K + img_size=960 (current) | 2.7 | High (equirect) | Simple | $0 (owned) |
| Insta360 X5 5.7K + img_size=1920 (Phase 0) | 5.3 | High (equirect) | Simple | $0 |
| Insta360 X5 + 3 virtual crops (Phase 1-2) | 12.8 | Low (perspective) | Simple | $0 |
| Dual DJI/GoPro 4K flat (Phase 3) | 38.0 | Near zero | Moderate | ~$700-800 |
| Veo Cam 3 (reference) | ~38.0 | Near zero | Simple (cloud) | $1,000+/yr |

---

## Design Principles

1. **Merge detections, not videos** — fuse ball detections from multiple views into one equirect-space track, don't stitch rendered videos
2. **Preserve the old path** — new detection modes gated behind config flags, current single-equirect path always available
3. **Same output contract** — `detections.jsonl` format unchanged regardless of detection source, downstream pipeline untouched
4. **Validate before building** — each phase has a decision gate based on measured improvement
5. **Training data compounds** — every processed game generates hard frames for labeling, each re-train improves the next run
