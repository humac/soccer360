# Soccer360 — Agent Workflow Context (Gemini)

## Current truth
Soccer360 already implements a real ball-first processing pipeline and operator dashboard.

### Implemented
- ingest watcher + dedupe
- `soccer360 watch/process/train/dashboard/export-hard-frames`
- detection/tracking/stabilization pipeline
- player-cluster / center-of-play support
- broadcast + tactical renders
- heuristic highlight clip export
- active-learning hard-frame export
- training flow and dashboard training/admin endpoints

### Not yet implemented
- `highlights.json`
- `highlights_reel.mp4`
- delivery-pack CLI
- person tracking/event/hybrid-camera outputs
- overlays

## Next milestone
**Ball-First Delivery Pack**

### Scope
- repair Label Studio pre-annotation compatibility
- add highlight manifest output
- add single highlight reel output
- document the real operator delivery flow

### Non-scope
- person detection/tracking
- `events.json`
- `broadcast_action.mp4`
- UI redesign

## Working rule
Stay anchored to existing code paths. Do not infer roadmap features as already shipped.

## Vision
Vision is **not needed** for this milestone.
