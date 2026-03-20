# Soccer360 AI Agent Prompt Pack
## Repo-Aware Planning Prompts for Multi-View Evolution

> Internal prompt/design aid for engineering and agent-guided planning.
> This document is not the current source of truth for product behavior, operator workflow, or shipped functionality.
> It should not be treated as a committed roadmap.

This version is updated with repo-aware guidance based on the current public `humac/soccer360` repository.

Verified current implementation references:
- `src/pipeline.py` orchestrates the phase flow and job lifecycle.
- `src/reframer.py` already performs 360-to-perspective rendering for broadcast and tactical outputs.
- The README documents the current end-to-end architecture and module layout.

Key current pipeline from the repo:
`360 video -> Detection (GPU) -> FoI Filter -> Tracking -> Player Cluster -> Camera Path -> Reframing -> Export`

Important repo-aware architecture facts:
- `Pipeline.run()` currently performs:
  - detection
  - tracking / stabilization
  - hard frame export
  - optional player cluster
  - camera path generation
  - broadcast reframing
  - tactical reframing
  - highlights
  - export
- `Reframer` already knows how to render perspective views from the equirectangular source using `py360convert.e2p`.
- This means the best path is to evolve the front of the pipeline, not rewrite the entire stack.

This prompt pack is designed to help a coding agent plan and implement a **manual-calibration-first, multi-view detection evolution** of the existing codebase.

---

## Repo Ground Truth Summary

Use these facts as grounding assumptions unless code inspection shows otherwise:

- `src/pipeline.py` is the main orchestration layer and the safest insertion point for new stages.
- `src/reframer.py` is the natural place to reuse or extend for 3 virtual detection-view generation.
- Current outputs should remain intact:
  - `broadcast.mp4`
  - `tactical_wide.mp4`
  - detections / tracks / player cluster / metadata / highlights artifacts
- Current player cluster and camera path logic should be preserved where possible and adapted to consume better upstream ball data instead of being rewritten.

---

## Master Planning Prompt (Repo-Aware)

```text
You are acting as the lead architect and implementation planner for the Soccer360 repository: humac/soccer360.

Grounding from the current repo:
- The existing pipeline is orchestrated in src/pipeline.py.
- The current phase order is:
  detection -> tracking/stabilization -> hard frame export -> optional player cluster -> camera path -> broadcast reframing -> tactical reframing -> highlights -> export
- The current reframing logic already exists in src/reframer.py and uses py360convert.e2p to render perspective output from the equirectangular source.
- The current detector stack is YOLO26l-based and the system is offline batch, not real-time.
- The current system processes the full 360 source and uses player clustering to support center-of-play and camera movement.
- The current outputs include broadcast, tactical, and highlights.

Target evolution:
We want to improve ball detection and downstream rendering quality by adding a new front-end path:
1. raw 360 input
2. calibration frame extraction
3. manual-assisted field calibration (primary v1 path)
4. optional auto field inference later
5. 3-view planner
6. generation of 3 virtual detection views
7. YOLO26l detection on each view
8. fusion into one global ball track
9. feed fused track into existing downstream pipeline components
10. preserve current outputs

Your planning goals:
- Review the actual repo code first
- Explain the current implementation in plain English
- Propose the smallest practical evolution path
- Reuse existing modules where possible
- Avoid a rewrite
- Preserve backward compatibility with the current single-360 path

Deliverables required:
1. Current-state implementation review tied to real files and functions
2. Future-state target architecture
3. A minimal-change migration plan
4. A file-by-file impact plan
5. Config and artifact contract changes
6. Risks and rollback strategy
7. Test strategy
8. Recommended phase-by-phase implementation order

Be explicit about:
- where to insert manual calibration
- where to insert 3-view planning
- whether to extend src/reframer.py or add a sibling module for detection-view generation
- how to preserve the old path behind a flag
- how current player clustering should still be used
```

---

## Stage 0 Prompt: Current-State Functional Review

```text
Review the Soccer360 repository and document the current pipeline as implemented, not as imagined.

Focus first on:
- src/pipeline.py
- src/detector.py
- src/tracker.py
- src/player_cluster.py
- src/camera.py
- src/reframer.py
- src/highlights.py
- src/exporter.py
- configs/pipeline.yaml

I want a functional implementation review, not a style review.

For each stage in src/pipeline.py, document:
1. phase name
2. owning class/function/module
3. input artifacts
4. output artifacts
5. key config dependencies
6. downstream consumers
7. extension points for the future multi-view pipeline

Then produce:
- a plain-English walkthrough of the current pipeline
- a mermaid sequence diagram
- a list of exact insertion points for:
  - calibration frame extraction
  - manual field calibration
  - 3-view planning
  - view export / view streaming
  - multi-view fusion

Do not recommend rewrites before mapping the existing code.
```

---

## New Primary Design Principle

```text
Design principle:
Do NOT merge 3 rendered videos back into one result as the source of truth.

Instead:
- generate 3 detection-friendly virtual views
- run detection on each
- fuse detections into one global ball track
- feed that fused track into the existing downstream camera/render stack

Merge detections, not videos.
```

---

## Manual Calibration First Addendum (Repo-Aware)

For this repo, manual-assisted field calibration should be treated as the recommended v1 path because:

- the camera is static for the match
- the current pipeline is already sophisticated enough without adding brittle auto-field inference first
- a one-time calibration step will likely produce better view planning and downstream stability faster than a fully automatic attempt

Target architecture addition:
```text
raw 360 input
-> representative calibration frame
-> manual field calibration metadata
-> 3-view planner
-> virtual detection views
-> YOLO26l per-view detection
-> fusion
-> existing track/camera/render/highlight pipeline (adapted)
```

---

## New Prompt: Manual Field Calibration Stage

```text
Design a manual-assisted field calibration stage for the current Soccer360 repository.

Repo context:
- src/pipeline.py is the orchestrator.
- src/reframer.py already renders perspective views from the source 360 video.
- We want a reliable v1 path before building full auto-calibration.

Goal:
Add a stage that extracts a representative frame from the raw 360 video, allows a user to define the playable field geometry once, stores that geometry as metadata, and uses it to drive the 3-view planner.

I want:
1. The smallest practical implementation path in this repo
2. Whether this should be:
   - a lightweight local web UI
   - a dashboard-integrated screen
   - or a CLI-assisted step with saved JSON
3. A proposed calibration metadata format
4. Whether to use:
   - polygon points
   - semantic anchor points
   - or a hybrid
5. How to integrate this into src/pipeline.py without breaking the current flow
6. How to keep the current no-calibration / old path available behind config
7. How to validate and debug the calibration visually

Please recommend the best v1 path for this specific repo, not a generic product.
```

---

## Stage 1 Prompt: Future-State Architecture for This Repo

```text
Based on the actual Soccer360 implementation, design the future-state architecture for a manual-calibration-first multi-view detection pipeline.

Constraints:
- keep the current repo structure as much as practical
- preserve current outputs
- preserve the old single-360 path behind a flag
- offline accuracy matters more than speed
- reuse current modules where practical

I want the design to answer:
1. What should be added to src/pipeline.py?
2. What should be added to or around src/reframer.py?
3. Should the 3-view detection-view export live in src/reframer.py or a new sibling module?
4. How should the fused global ball track be represented?
5. Which current modules should remain mostly unchanged?
6. Which current modules should be adapted?
7. What should be deferred to v2?

Include:
- data flow
- artifact flow
- config additions
- backward compatibility strategy
```

---

## Stage 2 Prompt: 3-View Planner

```text
Design the 3-view planner for the current Soccer360 repo.

Inputs:
- manual calibration metadata from a representative frame
- optionally future auto-field inference data
- source video metadata

Outputs:
- a stored 3-view plan for the match / half / segment
- parameters needed by the reframe/export stage

The planner should choose:
- center main view
- left support view
- right support view

Optimize for:
- field coverage
- likely play coverage
- reduced distortion
- better ball pixel size
- enough overlap for recovery, but not excessive redundancy

I want:
1. A practical scoring-based design
2. Candidate parameters:
   - yaw
   - pitch
   - FOV
3. A recommendation for v1:
   - fixed full match
   - per half
   - or per segment
4. A proposed metadata format for the selected plan
5. A debug/visualization plan
6. The cleanest integration point in src/pipeline.py
```

---

## Stage 3 Prompt: Detection-View Generation

```text
Design how the Soccer360 repo should generate the 3 detection views.

Repo context:
- src/reframer.py already renders perspective views from a 360 source for broadcast and tactical outputs.
- We likely want to reuse some of that machinery.

Question to solve:
Should we:
A. extend src/reframer.py to support generating 3 virtual detection views
B. create a sibling module dedicated to detection-view generation
C. provide a shared lower-level helper used by both

I want:
1. A recommendation for this repo
2. A file-by-file change plan
3. Whether views should be:
   - temporary frame streams
   - cached image sequences
   - cached videos
   - or another structure
4. How to preserve frame/time alignment across all 3 views
5. How to support reruns and debugging
6. How to avoid generating dead-end artifacts

Important:
The output of this stage should support downstream detection and later fusion into one global ball track.
```

---

## Stage 4 Prompt: Multi-View YOLO Detection

```text
Design how to adapt the current detector stack in src/detector.py for 3-view detection.

Requirements:
- keep YOLO26l
- maximize accuracy
- avoid introducing a second unrelated detector pipeline
- preserve the current single-view path

I want:
1. A plan to reuse the current detector implementation
2. How to represent per-view detections
3. Whether detections should be written as:
   - one file per view
   - one combined file with view_id
   - or another format
4. How to keep player detections useful for later player cluster logic
5. How to compare view-level performance
6. What config changes are required

Also explain:
- whether player clustering should remain downstream only
- or whether person detections from the 3 views can still help recovery / support confidence
```

---

## Stage 5 Prompt: Fusion Into One Global Ball Track

```text
Design the fusion stage for the current Soccer360 repo.

Goal:
Take per-view detections from the 3 detection views and produce one fused global ball track that can drive the existing downstream camera/render logic.

I want:
1. A practical v1 global coordinate representation
2. A mapping strategy from per-view detections into that global representation
3. How to resolve:
   - conflicting detections
   - duplicate detections
   - missing detections
   - false positives
4. Where to reuse existing tracking/stabilization logic if possible
5. How current player cluster logic can support fallback when fused ball confidence is weak
6. The output artifact contract

Important:
The output should be consumable by current or lightly adapted downstream modules.
```

---

## Stage 6 Prompt: Adapting Downstream Modules

```text
Plan how to adapt the downstream Soccer360 modules to use the fused ball track.

Focus on:
- src/tracker.py
- src/player_cluster.py
- src/camera.py
- src/reframer.py
- src/highlights.py
- src/exporter.py

For each module, answer:
1. Can it remain unchanged?
2. Does it need a new input contract?
3. Does it need an adapter layer?
4. Should the old single-360 path remain available?

The goal is to preserve as much as possible while improving the quality of the upstream ball signal.
```

---

## Stage 7 Prompt: Config and Artifact Contracts

```text
Create a repo-aligned config and artifact plan.

I want proposed additions to configs/pipeline.yaml for:
- calibration
- view_planner
- detection_views
- multiview_detection
- fusion

Also define:
- intermediate artifacts
- naming conventions
- cache / rerun behavior
- backward compatibility strategy
- whether these artifacts belong in scratch, processed output, or both

Keep this aligned with the current Soccer360 storage and export patterns.
```

---

## Stage 8 Prompt: Testing and Rollout

```text
Create a test and rollout plan for adding manual-calibration-first multi-view detection to Soccer360.

I want:
1. unit tests
2. integration tests
3. regression tests against the current pipeline
4. golden artifact tests if appropriate
5. a phased rollout plan
6. rollback strategy

Test scenarios should include:
- off-center camera placement
- crowded midfield
- far-away ball
- motion blur
- indoor field
- weak field line visibility
- missing or poor calibration data

The rollout should preserve the current single-360 path as a fallback until the new mode is proven.
```

---

## First Implementation Prompt (Repo-Aware)

```text
Implement only the first approved phase of the multi-view evolution for Soccer360.

Rules:
- do not start later phases
- preserve current pipeline behavior by default
- use flags/config to gate the new behavior
- keep the repo runnable
- do not refactor unrelated code
- add docs and tests with the implementation

Phase scope should be limited to the first approved milestone, likely one of:
- architecture and code mapping documentation
- calibration metadata scaffolding
- calibration frame extraction
- or another smallest safe slice approved in the plan

Required output:
1. files changed
2. what was added
3. what remains stubbed
4. how to run it
5. how to validate it
```

---

## Updated Ground Rules for the Coding Agent

```text
Ground rules:
- Evolve the existing Soccer360 pipeline. Do not rewrite it.
- Use src/pipeline.py as the main orchestration insertion point.
- Reuse or extend src/reframer.py where practical for virtual detection views.
- Preserve current outputs and current single-360 behavior by default.
- Prefer manual-assisted field calibration for v1.
- Keep future auto-calibration possible, but do not depend on it in v1.
- Keep YOLO26l as the detector.
- Merge detections, not videos.
- Treat the fused global ball track as the source of truth for downstream rendering.
- Reuse current player-cluster logic as a support signal where helpful.
- Keep every phase small enough to review safely.
- Leave the repo in a runnable state after each phase.
```

---

## Recommended Planning Sequence

1. Stage 0 — current-state functional review
2. Manual Calibration Stage
3. Stage 1 — future-state architecture
4. Stage 2 — 3-view planner
5. Stage 3 — detection-view generation
6. Stage 4 — multi-view YOLO detection
7. Stage 5 — fusion
8. Stage 6 — downstream adaptation
9. Stage 7 — config/contracts
10. Stage 8 — testing/rollout
11. First implementation prompt

---

## What Good Looks Like in This Repo

A good plan should result in:
- a repo-specific map of the current phase flow
- a small, safe insertion of manual calibration and 3-view planning into src/pipeline.py
- reuse of current 360 perspective rendering logic for detection-view generation
- YOLO26l reused across 3 virtual views
- one fused ball track feeding the existing downstream broadcast/tactical/highlight machinery
- preservation of the current path as fallback

---

## Dual-Path Addendum: CLI and Dashboard-Assisted Workflows

The new multi-view evolution must explicitly support **both** existing operating paths in Soccer360:

1. **CLI / headless processing path**
2. **Dashboard-assisted path**

Repo grounding:
- The README documents CLI commands such as `soccer360 watch`, `soccer360 process`, `soccer360 dashboard`, `soccer360 train`, and `soccer360 export-hard-frames`.
- `src/pipeline.py` already supports event-bus-assisted decision hooks during processing.
- `src/dashboard.py` already provides REST endpoints, SSE event streaming, staging/import workflows, inference model selection, training flows, media review, and processed-match reset behavior.

This means the new calibration and multi-view planning features must not be designed as dashboard-only or CLI-only hacks.
They must share the same backend contracts and pipeline behavior.

### Required design rule

```text
One backend pipeline
+ one calibration artifact format
+ one shared multi-view plan format
+ two input surfaces:
  - dashboard UI
  - CLI / file-based / automation path
```

### Recommended split of responsibility

#### Dashboard-assisted path
This should be the **primary v1 interactive experience** for manual calibration.

Recommended flow:
1. user uploads or stages a source video
2. user selects the video in the dashboard
3. system extracts a representative calibration frame
4. dashboard shows the frame
5. user draws field polygon or anchor points
6. system saves calibration metadata
7. system optionally previews proposed 3 virtual camera views
8. user starts processing
9. pipeline runs using the same backend artifacts the CLI would use

#### CLI / headless path
This should support non-interactive or repeatable processing.

Recommended flow:
- `soccer360 process input.mp4 --calibration path/to/calibration.json`
- or `soccer360 process input.mp4 --use-existing-calibration venue_or_match_key`
- or `soccer360 process input.mp4 --require-calibration`
- or config-driven fallback to the old single-360 path when calibration is missing

CLI should not be forced to provide an interactive drawing UI.
Instead, it should consume the same saved calibration artifact produced by the dashboard or another prep tool.

### Shared backend contract

Both paths must use:
- the same calibration metadata format
- the same 3-view planner
- the same detection-view generation
- the same fusion logic
- the same downstream rendering contracts

The dashboard should be an interaction layer.
The CLI should be an automation layer.
Neither should fork the core logic.

### Practical v1 recommendation for this repo

For v1:
- **Dashboard path** should be the primary place for manual field calibration UI
- **CLI path** should support supplying, requiring, reusing, or skipping calibration through flags/config
- the old path should remain available behind config / flags until the new path is proven

---

## New Prompt: Dual-Path UX and Pipeline Design

```text
Design the new Soccer360 multi-view calibration and planning flow so it works cleanly for both:
1. CLI / headless processing
2. dashboard-assisted interactive processing

Repo context:
- src/pipeline.py is the shared orchestrator
- src/dashboard.py already supports interactive operational workflows
- CLI commands are already part of the existing app model

I want:
1. A clear dual-path design that avoids duplicating backend logic
2. A shared backend contract for:
   - calibration artifacts
   - 3-view plans
   - multiview detection outputs
   - fused track outputs
3. A recommendation for how the dashboard should support manual calibration in v1
4. A recommendation for how the CLI should support:
   - explicit calibration file input
   - calibration reuse
   - require-calibration behavior
   - fallback behavior
5. A file-by-file impact analysis for both paths
6. A backward compatibility strategy so the current workflow keeps working

Please make sure the design treats:
- the dashboard as the main interactive UX
- the CLI as the main automation/headless UX
- the pipeline backend as shared logic
```

---

## New Prompt: CLI Path Planning

```text
Design the CLI path for the new Soccer360 calibration-first multi-view pipeline.

I want:
1. Proposed CLI flags / commands for:
   - supplying calibration
   - requiring calibration
   - reusing saved calibration
   - generating a calibration frame without full processing
   - optionally previewing the 3-view plan
2. How the CLI should behave when calibration is missing
3. How the CLI should preserve backward compatibility with the current process/watch flows
4. Whether watcher-based ingest should:
   - skip to old path
   - pause for dashboard calibration
   - or use configurable behavior
5. A minimal implementation path that keeps automation practical

Keep the design aligned with the current `soccer360 process` and `soccer360 watch` model.
```

---

## New Prompt: Dashboard Path Planning

```text
Design the dashboard-assisted path for the new Soccer360 calibration-first multi-view pipeline.

Repo context:
- src/dashboard.py already has routes for staging/import, processed media, training, settings, and operational decisions.
- We want the dashboard to become the primary v1 UX for manual field calibration.

I want:
1. The best place in the dashboard flow to insert calibration
2. Whether calibration should occur:
   - before ingest
   - after staging but before processing
   - or as a pause/decision point during processing
3. Proposed API endpoints and backend contracts for:
   - extracting a calibration frame
   - saving calibration metadata
   - previewing the 3-view plan
   - launching a run with calibration
4. A recommendation for the smallest practical UI implementation
5. How to keep the old dashboard workflow available during rollout
6. How this should interact with staging, watcher processing, and manual reprocessing

Keep the design practical for the existing FastAPI + static UI approach in this repo.
```

---

## Updated Ground Rules

```text
Additional ground rules:
- Explicitly support both CLI/headless and dashboard-assisted workflows.
- Do not fork core pipeline behavior between the two paths.
- The dashboard should be the primary v1 manual calibration UX.
- The CLI should support consuming saved calibration artifacts and automation-friendly flags.
- Use one shared calibration artifact format and one shared multiview backend path.
- Preserve current watcher/process/dashboard workflows until the new path is proven.
```

---

## Updated Planning Sequence

1. Stage 0 — current-state functional review
2. Dual-Path UX and Pipeline Design
3. Manual Calibration Stage
4. CLI Path Planning
5. Dashboard Path Planning
6. Stage 1 — future-state architecture
7. Stage 2 — 3-view planner
8. Stage 3 — detection-view generation
9. Stage 4 — multi-view YOLO detection
10. Stage 5 — fusion
11. Stage 6 — downstream adaptation
12. Stage 7 — config/contracts
13. Stage 8 — testing/rollout
14. First implementation prompt
