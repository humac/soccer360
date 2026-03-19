# DECISIONS

## 2026-03-19 — Normalize Soccer360 around the real implemented product
**Status:** Accepted

### Decision
Treat Soccer360 as an implemented **ball-first 360 video processing system with operator tooling**, not as an action-aware or player-aware product.

### Why
The codebase already contains substantial working functionality:
- ingest watcher + dedupe
- CLI processing commands
- normal and no-detect pipeline modes
- center-of-play/player-cluster support
- monitoring dashboard + training/admin endpoints
- active-learning exports and model training
- broadcast/tactical renders and heuristic highlight clips

But several roadmap docs describe future capabilities that are not in the repo today:
- person detection/tracking pipeline
- heuristic event system with `events.json`
- hybrid action-camera render
- highlight reel assembly and ranking artifacts
- overlays / parent-facing polish features

### Consequence
All workflow docs must distinguish:
- **current scope** = implemented ball-first pipeline and operator dashboard
- **future scope** = player/action understanding and richer product packaging

---

## 2026-03-19 — Use one bounded next milestone instead of a broad “Delivery 1.5” jump
**Status:** Accepted

### Decision
The next milestone is **Ball-First Delivery Pack**, not player detection or hybrid action rendering.

### Included
- normalize Label Studio pre-annotation for current hard-frame manifests
- emit `highlights.json`
- emit `highlights_reel.mp4`
- document a standard operator delivery flow using current pipeline outputs

### Excluded
- person detection/tracking
- `events.json`
- `broadcast_action.mp4`
- overlays
- dashboard redesign
- new action-recognition ML work

### Why
This milestone fits the current implementation and can deliver immediate value without introducing a new perception stack.

---

## 2026-03-19 — Vision is not required for the immediate milestone
**Status:** Accepted

### Decision
Do not route the next milestone through Vision by default.

### Why
The bounded milestone is primarily:
- output packaging
- metadata/index generation
- labeling workflow cleanup
- operator documentation

It does not require a new screen flow or dashboard redesign to succeed.

### Revisit trigger
Bring Vision in when any of the following become in-scope:
- dashboard-led parent/operator delivery UX
- new match review screens
- media browsing / reel management UX
- major dashboard workflow changes beyond minor copy or controls

---

## 2026-03-19 — Highlight package generation belongs to the highlight subsystem
**Status:** Accepted

### Decision
Make `src/highlights.py` the ownership boundary for:
- per-clip export
- `highlights.json`
- `highlights_reel.mp4`
- highlight-specific no-clips / reel-failure behavior

Keep `src/exporter.py` limited to final output preservation and high-level metadata/reference writing.

### Why
- highlight manifest and reel ordering depend on the same clip-selection result already computed in the highlight module
- exporter is currently a finalization layer, not a media-composition layer
- this avoids duplicate clip discovery logic and reduces source-of-truth drift

### Consequence
- no exporter-owned reel builder
- no exporter-owned manifest synthesis
- highlight directory contents are generated before export finalization, then copied/preserved as artifacts

---

## 2026-03-19 — Label Studio compatibility should be fixed at the import boundary
**Status:** Accepted

### Decision
Preserve the current hard-frame manifest export behavior for this milestone and repair schema compatibility in the Label Studio import path.

### Why
The real current issue is that the importer expects `predicted_bbox` while current active-learning manifests may emit `bbox`.
Fixing the importer is lower risk than changing the pipeline manifest contract everywhere first.

### Consequence
The Label Studio import helper must support both:
- `predicted_bbox` / `predicted_confidence`
- `bbox` / `conf`

This milestone does not require a broader manifest unification effort beyond that compatibility layer.

---

## 2026-03-19 — No separate packaging CLI command in this milestone
**Status:** Accepted

### Decision
Do not add a new `soccer360 pack` or equivalent command for Ball-First Delivery Pack.

### Why
- the normal `watch` / `process` flow already performs the full job lifecycle
- the gap is output completeness and documentation, not entrypoint coverage
- a second command would add duplicate orchestration and QA burden for little value

### Consequence
The delivery pack is produced, when available, as part of the standard processing flow.
