# Soccer360 — Product Roadmap (Delivery 1) + Coding Prompts

This roadmap is written to **ship a first “parent-ready” delivery** of the Soccer360 video pipeline, based on the current functional implementation (ball-first pipeline with reframed renders + highlight clips) and the known gaps (no players/actions yet, no overlays, Label Studio pre-annotation field mismatch).

Use this file as your **execution tracker**: check boxes as work lands.

---

## North Star: what “Delivery 1” means

### Delivery 1 output pack (per ingested match)
**Goal:** drop a raw 360 match video into `/tank/ingest/` → get a shareable “action-focused” pack.

**Minimum acceptable pack (Ball-First Parent Pack v1)**
- `/tank/processed/<game>/broadcast.mp4`
- `/tank/processed/<game>/tactical_wide.mp4`
- `/tank/highlights/<game>/highlight_###.mp4`
- `/tank/highlights/<game>/highlights_reel.mp4`  ✅ new
- `/tank/highlights/<game>/highlights.json`      ✅ new (index + why)

**Preferred pack (Action-Focused Parent Pack v1.5)**
Everything above, plus:
- `/tank/processed/<game>/broadcast_action.mp4`  ✅ new (hybrid camera)
- `/tank/processed/<game>/events.json`           ✅ new (heuristics)
- `/tank/processed/<game>/person_tracks.json`    ✅ new (players)

---

## Reality checks (don’t build on imaginary features)
- Today’s pipeline is **ball-first** (detect → stabilize → camera path → reframer → highlights → exporter).
- **No overlays** exist today.
- **Player detection/tracking + action/event understanding are not implemented yet**.
- Label Studio pre-annotation gotcha: V1 exporter uses `bbox` but LS import script expects `predicted_bbox` unless fixed.

---

## Milestones at a glance (ship in slices)

### M0 — Make the current pipeline “deliverable” (operator confidence)
- Focus: reliability, runbook, clear outputs, one-click-ish usage.

### M1 — Ball-First Parent Pack v1 (reel + index)
- Focus: one file parents will actually watch.

### M2 — Close the ball learning loop (Label Studio + train/promote)
- Focus: improve ball accuracy quickly and repeatably.

### M3 — Action-Focused Parent Pack v1.5 (players + events + hybrid camera)
- Focus: stop following empty grass when the ball disappears.

### M4 — (Optional polish) Overlays
- Focus: “broadcast feel” without new ML.

---

# M0 — Deliverable Pipeline Hardening

## M0 checklist
- [ ] M0.1 Add `soccer360 pack` (or `soccer360 deliver`) CLI command to produce a standard output pack for a given input.
- [ ] M0.2 Add a “golden match” smoke test script that processes one short clip and asserts expected artifacts exist.
- [ ] M0.3 Add a run summary in `metadata.json` (processing duration, highlight count, any NO_DETECT fallbacks).
- [ ] M0.4 Add a `docs/ops_runbook.md` with copy/paste commands for start/stop, single-run, logs, troubleshooting.

### M0.1 — CLI “pack” command
**Why:** operators (you) should run one command and get the exact files you expect.

**Implementation notes**
- Add command in `src/cli.py` (`soccer360 pack <video_path>`).
- Internally call the existing pipeline `Pipeline.run(...)` or reuse `soccer360 process`.
- Standardize output naming in a single place (avoid drift).

**Acceptance criteria**
- Running:
  ```bash
  docker compose run --rm worker soccer360 pack /tank/ingest/match.mp4
  ```
  results in the standard output pack under `/tank/processed/<game>/` and `/tank/highlights/<game>/`.

**Coding prompt (copy/paste)**
```text
You have repo access. Implement a new CLI command that makes Soccer360 “deliverable”.

Task: Add `soccer360 pack <video>` which runs the normal pipeline and guarantees a standardized deliverable pack.

Requirements:
1) Command lives in src/cli.py and is registered under the existing Click group.
2) It must:
   - load config the same way as existing commands
   - run the pipeline (same as watch/process)
   - print the final output directories (processed + highlights)
3) If the pipeline ran in NO_DETECT mode, the command must print a loud warning and set non-zero exit code.
4) Add a short docs entry in docs/ops_runbook.md with copy/paste commands.

Hard rules:
- Do not refactor unrelated code.
- Prefer calling existing pipeline surfaces (Pipeline.run / existing process command).

Deliver:
- Updated CLI
- Updated docs
- A minimal unit test that asserts the command exists (Click invocation test).
Include a Mermaid flowchart of the operator flow.
```

---

### M0.2 — Golden smoke test script
**Why:** You need a 60-second “did I break the pipeline?” check before spending hours processing a full match.

**Implementation notes**
- Add `scripts/smoke_test.sh`:
  - processes a short sample clip (or a small test asset)
  - asserts these exist: `broadcast.mp4`, `tactical_wide.mp4`, `metadata.json`
- Optionally add `scripts/smoke_test.py` for better checks (JSON validation).

**Acceptance criteria**
- `bash scripts/smoke_test.sh` exits `0` and prints output paths.

**Coding prompt**
```text
Add a golden-path smoke test for Soccer360.

Tasks:
1) Create scripts/smoke_test.sh that runs:
   docker compose run --rm worker soccer360 process <small_test_video>
2) After run, assert required outputs exist:
   - /tank/processed/<game>/broadcast.mp4
   - /tank/processed/<game>/tactical_wide.mp4
   - /tank/processed/<game>/metadata.json
3) Also assert that metadata.json contains mode="normal" unless allow_no_model enabled.
4) Fail fast with clear error messages.

If no test video exists in repo, create a small synthetic 2–3s mp4 in tests/assets using ffmpeg
and commit it (keep it tiny).

Deliver:
- scripts/smoke_test.sh
- tests/assets/<tiny_video>.mp4 (if needed)
- docs update: how to run smoke test
```

---

# M1 — Ball-First Parent Pack v1 (Highlight reel + index)

## M1 checklist
- [ ] M1.1 Generate `highlights.json` (ranked list of clips + why they were selected).
- [ ] M1.2 Build a single `highlights_reel.mp4` by concatenating clips.
- [ ] M1.3 Add config switches for reel length and selection rules (top N, min spacing).
- [ ] M1.4 Ensure exporter preserves `highlights.json` alongside existing clips.

### M1.1 — `highlights.json` index
**Why:** you want a machine-readable record of what got exported, and you’ll need it for future scoring improvements.

**Schema**
```json
{
  "game": "match_stem",
  "source_broadcast": "broadcast.mp4",
  "clips": [
    {
      "id": "highlight_003",
      "path": "highlight_003.mp4",
      "start_sec": 123.4,
      "duration_sec": 8.0,
      "signals": ["speed", "goal_box"],
      "score": 0.82
    }
  ]
}
```

**Acceptance criteria**
- A `highlights.json` appears in `/tank/highlights/<game>/` and lists every exported clip.

**Coding prompt**
```text
Add a highlights index file.

Tasks:
1) In src/highlights.py, after clips are detected and exported, write highlights.json
   into the same output directory as the highlight_###.mp4 files.
2) Include per-clip:
   - id, filename/path, start_sec, duration_sec
   - which signals fired (speed, goal_box, direction_change)
   - a simple score (even a weighted sum) so future ranking has a slot.
3) Update exporter (src/exporter.py) to preserve/copy highlights.json into /tank/highlights/<game>/.

Constraints:
- Do not change the existing highlight extraction behavior yet.
- Add a unit test with a synthetic tracks input that produces deterministic clip events.

Deliver:
- highlights.json generation
- tests
- brief doc section in docs/ops_runbook.md on where to find it
```

---

### M1.2 — `highlights_reel.mp4` builder
**Why:** Parents want one file.

**Implementation notes**
- Add `src/reel.py` (or keep inside highlights module) that:
  - takes list of clip paths
  - writes `concat.txt`
  - runs ffmpeg concat demuxer to create `highlights_reel.mp4`

**Acceptance criteria**
- Reel exists in `/tank/highlights/<game>/highlights_reel.mp4`
- Plays cleanly, no missing audio/video streams.

**Coding prompt**
```text
Implement highlight reel assembly.

Tasks:
1) Add a reel builder that concatenates highlight clips into highlights_reel.mp4.
2) Use ffmpeg concat demuxer (safe, no re-encode if possible; re-encode if stream params mismatch).
3) Integrate into the pipeline:
   - After highlight clips are exported, generate the reel.
4) Update metadata.json (processed output) to include:
   - highlight_clip_count
   - highlight_reel_path (relative)
5) Add config:
   highlights:
     reel_enabled: true
     reel_top_n: 15
     reel_max_minutes: 6

Deliver:
- code + config wiring
- tests that mock ffmpeg call and assert concat list correctness
```

---

# M2 — Close the Ball Learning Loop (Label Studio + train/promote)

## M2 checklist
- [ ] M2.1 Fix Label Studio pre-annotation contract (`bbox` vs `predicted_bbox`).
- [ ] M2.2 Add `soccer360 train-ball-from-ls` command (export → dataset → train → promote).
- [ ] M2.3 Add training metrics artifact (`ball_best.metrics.json`) and version log.
- [ ] M2.4 Add a “model eval on a reference match” command (`soccer360 eval-ball`).

### M2.1 — Fix Label Studio pre-annotation mismatch
**Why:** without this, your “hard frames” import shows no boxes and labeling becomes slower and more error-prone.

**Two minimal fixes (choose one)**
- Option A: update `scripts/labelstudio_import.sh` to accept `bbox` OR `predicted_bbox`.
- Option B: update V1 exporter to also emit `predicted_bbox` alongside `bbox`.

**Acceptance criteria**
- Importing tasks for a new match shows pre-drawn rectangles in Label Studio.

**Coding prompt**
```text
Fix Label Studio pre-annotation.

Context:
- V1 hard_frames.json uses `bbox`.
- labelstudio_import.sh expects `predicted_bbox`.
This can cause no pre-annotations.

Tasks:
1) Update scripts/labelstudio_import.sh to:
   - prefer predicted_bbox if present
   - fallback to bbox if present
   - convert bbox_xyxy -> LS percent rectangle (x,y,width,height)
2) Add a tiny python helper module if bash becomes too messy:
   - scripts/ls_tasks.py with a CLI wrapper
3) Add a test fixture hard_frames.json (both schemas) and assert tasks.json contains predictions.

Deliver:
- working pre-annotation for both schemas
- docs snippet: how to import tasks
```

---

### M2.2 — `train-ball-from-ls` command
**Why:** you want a loop that ends with `/tank/models/ball_best.pt` updated and ready for next run.

**Acceptance criteria**
- One command takes LS export → produces dataset → trains → promotes best weights → writes metrics JSON.

**Coding prompt**
```text
Implement an operator-friendly ball training loop.

Add: `soccer360 train-ball-from-ls --ls-export <path> --out /tank/labeling/dataset --model-out /tank/models/ball_best.pt`

Requirements:
1) Parse Label Studio export JSON and map each labeled rectangle to YOLO format:
   class 0 = ball
2) Write labels into /tank/labeling/<match>/labels/ and ensure images are discoverable.
3) Reuse or call scripts/build_dataset.sh logic (prefer moving logic into python module for testability).
4) Call existing training logic (src/trainer.py or existing scripts) and promote best model to ball_best.pt
5) Emit:
   - /tank/models/ball_best.metrics.json (mAP, precision, recall if available)
   - /tank/models/versions.json (append-only: timestamp, git sha if possible, metrics)

Tests:
- Unit test label conversion from LS rectangle -> YOLO normalized coords.

Keep it minimal. No major refactors.
```

---

# M3 — Action-Focused Parent Pack v1.5 (Players + events + hybrid camera)

## M3 checklist
- [ ] M3.1 Add person detection (`person_detections.jsonl`) with config gates.
- [ ] M3.2 Add person tracking (`person_tracks.json`) (multi-object).
- [ ] M3.3 Add heuristic event detector (`events.json`).
- [ ] M3.4 Add hybrid camera path generator (`camera_path_action.json`).
- [ ] M3.5 Render `broadcast_action.mp4` using hybrid camera path.
- [ ] M3.6 Feed events into highlight scoring (simple weighted add-on, no ML yet).

> Note: This stage is the “real” jump from ball-only to action-aware without taking on full action-model training yet.

---

### M3.1 — Person detection (COCO class 0) as a parallel stream
**Why:** “where the play is” is mostly “where the people are”, especially when the ball disappears.

**Implementation notes**
- Add `src/person_detector.py` (can be a thin wrapper around existing Detector patterns).
- Config section `person_detection:` (enabled, model_path, conf, iou, img_size, max_det).
- Artifact: `{work_dir}/person_detections.jsonl` → preserved to processed output.

**Acceptance criteria**
- Processing a match produces `person_detections.jsonl` with class_id 0 boxes on many frames.

**Coding prompt**
```text
Add person detection without disturbing the ball path.

Tasks:
1) Create src/person_detector.py with API:
   - __init__(config)
   - run_streaming(video_path, meta, output_path) -> frames_processed
2) Use YOLO with COCO person class (0). Use a reasonable default model (yolov8m.pt or the baked yolov8s.pt).
3) Write person_detections.jsonl schema:
   {"frame_index": int, "time_sec": float, "bbox_xyxy":[x1,y1,x2,y2], "conf": float, "class_id": 0}
4) Wire into Pipeline.run:
   - Phase 1b after ball detection (or in parallel later)
5) Update exporter to preserve person_detections.jsonl.

Constraints:
- Do not add overlays.
- Do not modify ball detection behavior.
- Add a unit test using synthetic detections list for writer schema.
Include Mermaid flowchart showing new branch.
```

---

### M3.2 — Person tracking (multi-object)
**Why:** highlights and camera need stable motion signals, not noisy per-frame detections.

**Implementation notes**
- Add `src/person_tracker.py` using ByteTrack-like association (you already have legacy ByteTrack code in `src/tracker.py`).
- Output: `person_tracks.json` with per-frame list of player tracks.

**Acceptance criteria**
- Stable track IDs persist across short occlusions.

**Coding prompt**
```text
Implement a first-pass PersonTracker.

Tasks:
1) Create src/person_tracker.py.
2) Reuse/adapt existing ByteTrack logic from src/tracker.py, but for multi-object persons.
3) Input: person_detections.jsonl
4) Output: person_tracks.json schema:
   [
     {"frame": 42, "players": [{"track_id": 7, "x": 120.0, "y": 260.0, "bbox":[...], "confidence": 0.86}]}
   ]
5) Wire into Pipeline.run as Phase 2b (after ball stabilization).

Add tests with small synthetic frames and bounding boxes that should produce 2 consistent tracks.

Keep it practical: no appearance re-id yet.
```

---

### M3.3 — Heuristic event detector (`events.json`)
**Why:** you need *some* notion of “action” before training action models.

**Events to implement (v1)**
- `possession_candidate`: nearest player to ball under threshold
- `congestion`: number of players within radius of ball > N
- `transition`: ball speed spike or direction change + nearby players
- `lost_ball`: ball missing for > K frames (already known)

**Acceptance criteria**
- `events.json` appears and is referenced by highlight scoring and/or camera logic.

**Coding prompt**
```text
Add src/event_detector.py to generate events.json from ball tracks + person tracks.

Tasks:
1) Implement EventDetector.run(ball_tracks_path, person_tracks_path, meta, output_path).
2) Compute at least:
   - possession_candidate (nearest player to ball)
   - congestion (players within radius)
   - transition (ball speed spike)
   - lost_ball streak markers
3) Write events.json schema:
   [{"frame": int, "time_sec": float, "type": str, "confidence": float, "players_involved":[int], "ball": {"x":float,"y":float}}]
4) Wire into Pipeline.run after person tracking.
5) Preserve events.json in exporter.

Tests:
- Use tiny synthetic tracks fixtures and assert event types appear.

No ML. Pure heuristics.
```

---

### M3.4 — Hybrid camera path (`camera_path_action.json`)
**Why:** ball-only camera whips to empty space when the ball disappears behind players.

**Hybrid targeting rule (v1)**
- If ball present/confident → center mostly on ball
- If ball lost/low confidence → center on action centroid (player cluster or possession candidate)
- Smooth with existing Kalman + EMA + pan clamp

**Acceptance criteria**
- A match with ball loss yields smoother, more watchable camera.

**Coding prompt**
```text
Implement hybrid camera path generation.

Tasks:
1) Add src/action_camera.py OR extend src/camera.py with a hybrid mode.
2) Inputs: ball tracks + person tracks + (optional) events.json
3) Compute an action centroid:
   - either densest player cluster near ball or average of players near possession_candidate
4) Blend:
   target = w_ball * ball + w_action * centroid
   w_ball depends on ball status/confidence.
5) Output camera_path_action.json (same per-frame schema as camera_path.json).
6) Update Reframer to render broadcast_action.mp4 using camera_path_action.json when enabled:
   camera:
     mode: hybrid
7) Add tests with synthetic tracks: ball disappears, centroid persists.

Deliver:
- camera_path_action.json
- broadcast_action.mp4
- config flags
- Mermaid sequence diagram: tracking -> events -> hybrid camera -> reframer
```

---

### M3.6 — Feed events into highlight scoring (simple add-on)
**Why:** highlights should prioritize “actual soccer moments”, not just ball jitter.

**Acceptance criteria**
- highlight selection changes when events exist; still works when events missing.

**Coding prompt**
```text
Enhance highlight scoring using events.json.

Tasks:
1) Extend HighlightDetector.detect_and_export to accept optional events_path.
2) If present, load events and:
   - increase score for clips near "transition" and high "congestion"
   - optionally create new candidate windows around "possession_candidate" changes
3) Keep existing speed/goal_box/direction logic as baseline fallback.
4) Update highlights.json to include event-derived signals.

Tests:
- synthetic event list causes at least one clip to be selected/ranked higher.

Keep it minimal and reversible via config:
highlights:
  use_events: true
```

---

# M4 — Optional polish: Overlays (boxes, trails, timestamps)

## M4 checklist
- [ ] M4.1 Add overlay module (OpenCV) to draw ball marker/trail.
- [ ] M4.2 Optional: draw player boxes + IDs.
- [ ] M4.3 Add timestamp + event label overlay.
- [ ] M4.4 Config switches to disable overlays for speed.

**Coding prompt**
```text
Add simple overlays to the reframed output.

Context: current renders are clean crops, no overlays.

Tasks:
1) Create src/overlay.py with functions:
   - draw_ball(frame, ball_pos, trail)
   - draw_players(frame, players)
   - draw_hud(frame, timestamp, event_label)
2) Integrate into src/reframer.py render loop:
   - after e2p conversion, before FFmpegFrameWriter
3) Config:
   render:
     overlays:
       enabled: true
       draw_people: false
       draw_ball_trail: true
       draw_timestamp: true
4) Performance:
   - keep it simple: cv2.circle, cv2.rectangle, cv2.putText

Deliver:
- broadcast_action_overlay.mp4 (or same output when enabled)
- tests: unit tests for overlay functions on dummy frames
```

---

# Execution Tracker (copy/paste into PR descriptions)

## “Done when…”
- [ ] A match dropped into `/tank/ingest/` produces **one reel** parents can watch (`highlights_reel.mp4`).
- [ ] The run is reproducible: processed output contains `config_snapshot.json`, `metadata.json`, and artifacts.
- [ ] Label Studio import shows pre-annotations (no “empty tasks” surprise).
- [ ] (Preferred) Hybrid action camera render exists (`broadcast_action.mp4`) and is visibly better than ball-only.

---

# Notes on staging / effort (keep yourself honest)

- **Fastest parent value**: M1 (reel + index) + M0 (pack command + smoke test).
- **Fastest accuracy win**: M2.1 + M2.2 (pre-annotation + one-command training loop).
- **Most noticeable upgrade**: M3.4 (hybrid camera), but it depends on person signals.

---

# Appendix — Config snippets (additive)

## Highlights reel config (proposed)
```yaml
highlights:
  enabled: true
  reel_enabled: true
  reel_top_n: 15
  reel_max_minutes: 6
  use_events: false   # becomes true in M3.6
```

## Person detection/tracking config (proposed)
```yaml
person_detection:
  enabled: true
  model_path: /app/models/yolov8m.pt
  classes: [0]
  conf: 0.40
  iou: 0.45
  img_size: 640
  max_det: 80
  device: "cuda:0"

person_tracker:
  enabled: true
  track_high_thresh: 0.4
  track_low_thresh: 0.2
  new_track_thresh: 0.4
  track_buffer: 60
  match_thresh: 0.3

events:
  enabled: true
  possession_distance_px: 80
  congestion_radius_px: 120
  congestion_min_players: 6
  transition_speed_px_s: 500
  lost_ball_frames: 15

camera:
  mode: ball   # ball|hybrid
  hybrid:
    w_ball_default: 0.8
    w_ball_lost: 0.2
    smoothing_alpha: 0.35
```
