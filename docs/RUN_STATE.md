# RUN_STATE

## Project Status
- Project: Soccer360
- State: Normalized into the team workflow model
- Date: 2026-03-19
- Owner: Pepper
- Current phase: Ready for architecture refinement and bounded build planning

## What just happened
- Reviewed the current repo, operator docs, roadmap docs, core runtime entrypoints, and test suite shape.
- Separated current implemented behavior from aspirational roadmap material.
- Created the standard workflow docs under `docs/agent-workflow/`.
- Defined one bounded next milestone based on repository reality.

## Current truth snapshot
- Implemented now: CLI commands for watch/process/train/dashboard/export-hard-frames; watcher-based ingest queue; two-pass broadcast+tactical rendering; highlight generation; dashboard; active-learning exports; training flow; persistent ingest dedupe; NO_DETECT fallback; center-of-play/player-cluster support; Dockerized worker/dashboard/Label Studio runtime.
- Not implemented now: parent-ready reel pack, `highlights_reel.mp4`, `highlights.json`, person detection/tracking artifacts, `events.json`, hybrid action-camera render, overlays, automated one-command end-to-end training loop from Label Studio export.

## Next bounded milestone
- Milestone: **Ball-First Delivery Pack**
- Objective: make the current ball-first system produce a cleaner operator/parent deliverable without adding new ML subsystems.
- Scope:
  1. fix Label Studio pre-annotation contract mismatch (`bbox` vs `predicted_bbox`)
  2. generate `highlights.json`
  3. generate `highlights_reel.mp4`
  4. expose/document a standard delivery/pack workflow around existing outputs
- Why this milestone:
  - builds directly on existing implemented pipeline surfaces
  - improves usefulness without inventing player/action intelligence
  - is small enough for one development cycle

## Vision involvement
- Vision required: **No** for the chosen milestone.
- Reason: the next milestone is backend/export/operator-flow oriented, not a new product surface or major UI redesign. Vision may become useful later if the dashboard becomes the primary delivery surface or if parent/operator UX is expanded.

## Handoff guidance
- Tony should deepen `docs/agent-workflow/ARCH.md` into an implementation plan for the bounded milestone.
- Peter should not begin feature work until Tony converts the milestone into concrete task slices.
- Heimdall QA should validate docs and runtime outputs only after implementation exists.
