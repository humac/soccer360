# Soccer360 — Agent Workflow Context (Claude)

## Normalized project identity
Soccer360 is currently a **ball-first 360 soccer processing pipeline with operator tooling**, not yet an action-aware player-tracking product.

## What is implemented now
- watch/process/train/dashboard/export-hard-frames CLI commands
- watcher-based ingest queue with persistent dedupe
- normal pipeline mode plus NO_DETECT fallback
- ball detection, stabilization/tracking, camera path generation
- optional player-cluster / center-of-play support
- broadcast render + tactical wide render
- heuristic highlight clip export
- active-learning hard-frame export
- dashboard for monitoring, staging, training, reset/requeue flows
- training/promotion path for the ball model
- broad pytest coverage across major modules

## What is not implemented now
- `highlights_reel.mp4`
- `highlights.json`
- dedicated `pack` delivery command
- separate person tracking artifact pipeline (`person_detections.jsonl`, `person_tracks.json`)
- `events.json`
- `broadcast_action.mp4`
- output overlays
- one-command Label Studio -> dataset -> train -> promote loop

## Next bounded milestone
**Ball-First Delivery Pack**

### In scope
1. fix Label Studio pre-annotation compatibility for current hard-frame manifests
2. add `highlights.json`
3. add `highlights_reel.mp4`
4. document a truthful operator delivery workflow

### Out of scope
- person detection/tracking
- action events
- hybrid camera
- overlays
- dashboard redesign

## Guidance for future Claude-style implementation work
- Do not “pull forward” roadmap features unless the task explicitly asks for them.
- Preserve the existing ball-first pipeline shape in `src/pipeline.py`.
- Prefer additive changes over architecture churn.
- Keep docs truthful: current implementation first, roadmap second.
- If touching output contracts, update tests and operator docs together.

## Vision involvement
Vision is **not required** for the current next milestone.
