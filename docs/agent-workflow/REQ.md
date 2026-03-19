# REQ — Soccer360

## 1. Project Definition
Soccer360 is currently a **ball-first automated 360 soccer video processing pipeline** with operator tooling.

Given an equirectangular match video, the system can currently:
- ingest and queue files from a watched directory
- detect ball and player/person objects in the detection pass
- stabilize ball tracking for camera guidance
- compute an optional player-cluster / center-of-play signal
- render a broadcast-style follow video and a tactical wide video
- export heuristic highlight clips
- export hard frames for labeling
- train/promote a ball model
- expose dashboard/operator controls for monitoring, staging, training, and reset/requeue flows

## 2. Current Implemented Scope

### 2.1 Implemented runtime surfaces
- CLI commands in `src/cli.py`:
  - `watch`
  - `process`
  - `train`
  - `export-hard-frames`
  - `dashboard`
- Docker services in `docker-compose.yml`:
  - `worker`
  - `labelstudio`
  - `dashboard`
  - `parent-site` (static Nginx site present in compose, not central to the core documented pipeline)

### 2.2 Implemented pipeline behaviors
- Two-pass processing orchestrated in `src/pipeline.py`
- Model resolution + normal / NO_DETECT mode handling
- Streaming frame processing via ffmpeg pipes
- Ball detection pipeline with FoI filtering, y-range filtering, best-per-frame selection
- Ball stabilization or legacy tracking path depending on config mode
- Active-learning hard-frame export
- Optional player-cluster computation for center-of-play support
- Camera path generation
- Broadcast render and tactical wide render
- Highlight clip detection/export
- Export of artifacts, metadata, and ingest archival status

### 2.3 Implemented operator/admin behaviors
- Watcher ingest dedupe persistence
- Dashboard job history / progress / SSE monitoring
- Dashboard training endpoints and staging import/reset flows
- Label Studio integration for manual annotation
- Training flow via `soccer360 train`
- Container/image verification flow documented in README and CLAUDE/GEMINI context files

### 2.4 Implemented test surface
The repo has broad pytest coverage across detector, stabilizer, camera, exporter, dashboard, events store, pipeline, tracker, trainer, watcher, and related modules. This indicates a meaningful implemented codebase rather than a doc-only prototype.

## 3. Explicitly Not Current Scope
The following appear in roadmap or future docs but are **not implemented as first-class current capabilities**:
- `highlights_reel.mp4`
- `highlights.json`
- `soccer360 pack` / standardized parent delivery command
- person detection/tracking artifact pipeline as separate exported deliverables (`person_detections.jsonl`, `person_tracks.json`)
- `events.json`
- hybrid action-camera output such as `broadcast_action.mp4`
- overlays/annotation rendering in output videos
- one-command Label Studio export -> dataset -> train -> promote operator loop
- mature parent-facing product/package flow beyond current raw outputs and dashboard/admin surfaces

## 4. Problem Statement For The Next Milestone
The current repository has a capable ball-first processing system, but its operator-facing output packaging is still fragmented:
- highlight clips exist, but no indexed highlight manifest exists
- no single highlight reel is assembled
- roadmap/docs describe a parent-ready pack that is not yet delivered by code
- Label Studio pre-annotation has a documented schema mismatch risk (`bbox` vs `predicted_bbox`)

This makes the current system useful to operators/engineers, but less consistent as a repeatable delivery workflow.

## 5. Next Bounded Milestone
## Milestone name
**Ball-First Delivery Pack**

## Milestone intent
Improve the usefulness and consistency of the existing ball-first system without introducing new ML domains.

## In scope
1. Fix the current hard-frame -> Label Studio pre-annotation compatibility issue.
2. Generate `highlights.json` alongside highlight clips.
3. Generate `highlights_reel.mp4` from selected highlight clips.
4. Define/document one standard delivery workflow for operators using current outputs.
5. Persist any resulting delivery metadata in the existing export structure.

## Out of scope
- person detection/tracking implementation
- action/event intelligence
- hybrid camera rendering
- overlays
- dashboard redesign
- parent portal buildout

## 6. Testable Requirements For The Next Milestone

### R1 — Label Studio pre-annotation compatibility
The system shall ensure the hard-frame labeling import flow works with the currently exported hard-frame manifest schema.

**Acceptance checks**
- A hard-frame manifest produced by the current pipeline can be converted into Label Studio tasks with pre-drawn ball rectangles.
- The import path supports the active manifest field naming used by current pipeline exports.
- Documentation identifies the supported manifest fields and expected operator workflow.

### R2 — Highlight manifest generation
The system shall emit a machine-readable highlight index for each processed match when highlight clips are produced.

**Acceptance checks**
- `highlights.json` is written for a processed match with highlights.
- It includes one entry per exported highlight clip.
- Each entry includes enough metadata to identify the clip and why it was selected.

### R3 — Highlight reel generation
The system shall generate a single highlight reel from the selected highlight clips.

**Acceptance checks**
- `highlights_reel.mp4` is written under the highlights output for a processed match when highlights are available.
- The reel is assembled from exported clips in a deterministic order.
- Failure behavior is defined when no clips are available.

### R4 — Standard delivery workflow
The repo shall document one truthful operator workflow for producing and locating the delivery pack.

**Acceptance checks**
- README and/or workflow docs describe where delivery assets appear.
- The workflow does not claim files that are not generated by the implementation.
- The workflow distinguishes required assets from optional/future assets.

### R5 — No feature invention
The milestone shall preserve the current ball-first architecture and not claim player/event/action capabilities that do not yet exist.

**Acceptance checks**
- No new requirements in this milestone depend on person tracking, event recognition, or hybrid action rendering.
- Docs remain aligned to current repo reality.

## 7. Vision Requirement
**Vision required for this milestone: No.**

Reason:
- the milestone is centered on export packaging, metadata, and operator workflow cleanup
- it does not require a new product screen set or a redesigned interactive dashboard flow

If the team later chooses to surface reels, curation controls, or match-review UX in the dashboard, Vision should then be involved.

## 8. Open Questions For Tony / Next Phase
- Where should reel assembly live: `src/highlights.py`, a new `src/reel.py`, or exporter-level orchestration?
- Should the delivery workflow be a new CLI command or a documented outcome of `process`/`watch` only?
- Should `highlights.json` live exclusively in the highlights directory, or also be referenced from processed-match metadata?
