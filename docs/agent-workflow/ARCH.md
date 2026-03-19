# ARCH — Soccer360

## Milestone
**Ball-First Delivery Pack**

## Purpose of this architecture pass
Define the smallest truthful architecture needed to make the current ball-first pipeline produce a cleaner operator delivery package.

This pass is intentionally bounded to:
1. Label Studio pre-annotation compatibility
2. `highlights.json`
3. `highlights_reel.mp4`
4. a truthful operator delivery workflow

It explicitly does **not** introduce:
- new perception systems
- player/action intelligence expansion
- new UI/dashboard product work
- speculative parent-facing packaging beyond the assets above

---

## 1. Current architecture baseline
The existing pipeline already has the right major shape for this milestone:

1. detect / stabilize tracks
2. export hard frames for labeling
3. generate camera path
4. render `broadcast.mp4`
5. render `tactical_wide.mp4`
6. export highlight clips into `work_dir/highlights/`
7. finalize outputs into `/tank/processed/<match>/` and `/tank/highlights/<match>/`

Relevant current ownership in code:
- `src/pipeline.py` orchestrates phase order
- `src/highlights.py` detects highlight-worthy moments and exports per-clip files
- `src/exporter.py` copies/moves final artifacts into stable output directories and writes `metadata.json`
- `scripts/labelstudio_import.sh` converts hard-frame manifests into Label Studio task JSON

That means this milestone should be solved as a **small extension of the existing highlight/export path**, not as a pipeline redesign.

---

## 2. Architecture decisions for this milestone

### Decision A — Label Studio compatibility should be fixed at the import boundary
**Decision:** Keep the hard-frame manifest contract produced by the pipeline intact for now, and make the Label Studio import path accept the active schema.

**Why:**
- the repo already emits active-learning manifests used by downstream tooling
- the current issue is specifically a compatibility mismatch between exported manifest fields and the import helper
- changing pipeline manifests first creates wider regression risk than fixing the importer

**Result:**
- `scripts/labelstudio_import.sh` or a small Python helper behind it must support both:
  - `predicted_bbox` / `predicted_confidence`
  - `bbox` / `conf`
- if both exist, importer should prefer the legacy explicit pre-annotation pair only if values differ by intent; otherwise either is acceptable as long as behavior is deterministic and documented

**Non-goal:** unify every historical manifest shape in the repo. Only the currently real shapes need support.

---

### Decision B — Highlight manifest and reel logic should live with highlight generation, not in exporter
**Decision:** `src/highlights.py` owns:
- per-clip export
- `highlights.json` generation
- `highlights_reel.mp4` assembly
- no-clips behavior for highlight outputs

`src/exporter.py` only owns final preservation/reference of those generated assets.

**Why:**
- the exporter should remain a file-finalization layer, not a media-editing layer
- highlight manifest contents depend on highlight ranking/selection context already present in `src/highlights.py`
- reel ordering should use the same clip-selection result that produced the clips
- keeping all highlight package generation in one module reduces duplicated logic and “source of truth” drift

**Consequence:**
- no new exporter-owned clip discovery logic
- no exporter-side inference about highlight score/order
- exporter copies the whole `work_dir/highlights/` directory as the canonical delivery bundle for highlight assets

---

### Decision C — No new CLI command is required for this milestone
**Decision:** Do not add `soccer360 pack` or another packaging command in this milestone.

**Why:**
- `soccer360 process` / watcher already executes the full job lifecycle
- the missing pieces are output completeness and documentation, not orchestration entrypoints
- a new command would create duplicate workflow surface area and extra QA cost

**Result:**
The delivery pack is the normal result of a successful `process`/`watch` run in normal mode.

---

### Decision D — `highlights.json` belongs beside the highlight clips
**Decision:** The canonical highlight manifest lives at:
`/tank/highlights/<match>/highlights.json`

**Why:**
- it describes the contents of that directory
- operators looking for highlight outputs should find the manifest next to the clips and reel
- keeping it in the highlight bundle avoids split-brain between processed metadata and highlight assets

**Secondary reference:**
`metadata.json` under `/tank/processed/<match>/` may include a pointer/reference to the highlight directory and key assets, but must not duplicate detailed clip records as a second source of truth.

---

### Decision E — Failure and no-clips behavior must be explicit and truthful
**Decision:** Distinguish three cases clearly:

1. **No highlight events / no clips selected**
   - no `highlights_reel.mp4`
   - no per-clip files
   - no `highlights.json`
   - `metadata.json` should still indicate that no highlight bundle was produced

2. **Highlight clips exist and reel succeeds**
   - write clips
   - write `highlights.json`
   - write `highlights_reel.mp4`

3. **Highlight clips exist but reel assembly fails**
   - preserve exported clip files
   - preserve `highlights.json`
   - do **not** pretend reel exists
   - surface reel failure in logs and, if practical, in metadata

**Why:**
- operators need truthful outputs, not silent partial success
- a reel generation failure should not discard useful clips already produced
- no-clips is a legitimate outcome in a ball-first heuristic pipeline, not necessarily a processing failure

---

## 3. Target artifact layout after this milestone

### Processed match directory
`/tank/processed/<match>/`
- `broadcast.mp4`
- `tactical_wide.mp4`
- `camera_path.json`
- `detections.jsonl`
- `tracks.json`
- `foi_meta.json`
- `hard_frames.json`
- `metadata.json`
- `ffprobe_meta.json`
- `config_snapshot.json`

### Highlight delivery directory
`/tank/highlights/<match>/`
- `highlight_000.mp4`, `highlight_001.mp4`, ...
- `highlights.json`
- `highlights_reel.mp4` *(only when clips exist and reel assembly succeeds)*

### Labeling directory
`/tank/labeling/<match>/`
- `frames/`
- `hard_frames.json`
- `labelstudio/tasks.json` *(manual import helper output)*

This milestone does **not** add new top-level directories or a second delivery bundle location.

---

## 4. Highlight manifest schema for this milestone
The schema should stay minimal and reflect what the current pipeline really knows.

Recommended shape:

```json
{
  "clip_count": 2,
  "reel_filename": "highlights_reel.mp4",
  "clips": [
    {
      "filename": "highlight_000.mp4",
      "rank": 2,
      "score": 3.5,
      "start_sec": 12.0,
      "end_sec": 18.0,
      "duration": 6.0,
      "event_types": ["speed", "goal_box"],
      "event_count": 3
    }
  ],
  "detector_stats": {
    "total_raw_events": 12,
    "speed_events": 4,
    "goal_box_events": 3
  }
}
```

Notes:
- `reel_filename` should be omitted or set to `null` when no reel exists
- manifest should not claim player IDs, event semantics, possession, team, or action labels
- clip order in `clips` should match actual exported filename order
- `rank` can continue to reflect score rank if sequential export remains time-sorted

---

## 5. Reel assembly design

### Placement
Reel assembly should be implemented inside `src/highlights.py`, preferably as a dedicated helper method or a small adjacent helper class/function, but still owned by the highlight subsystem.

### Input
- exported clip files already written into `work_dir/highlights/`
- deterministic ordered list of clip outputs from the same clip-selection pass

### Assembly method
Use ffmpeg concat demuxer or equivalent deterministic sequential concatenation.

### Required behavior
- reel order must be deterministic
- recommended order: the same order used for exported highlight filenames
- if zero clips exist, skip reel generation cleanly
- if one clip exists, reel may still be created as a concatenated single-clip output if implementation is simplest

### Failure policy
- reel creation failure should not fail the entire match export unless the current pipeline already treats highlight export as hard-fail
- preserve clips + manifest even if reel creation fails
- emit a clear log line

This is the key bounded quality-of-life improvement in the milestone: a single playable reel built from the clips the system already knows how to create.

---

## 6. Metadata ownership

### In `highlights.json`
Own the detailed highlight package description:
- clip filenames
- timing
- ranking/score
- detector/event summary
- reel presence/filename

### In `metadata.json`
Only reference highlight outputs at a high level:
- highlight output directory path
- maybe reel path if present
- maybe clip_count
- optional highlight status such as `none`, `clips_only`, `clips_and_reel`, `reel_failed`

This avoids making `metadata.json` a duplicate detailed manifest.

---

## 7. Operator workflow for this milestone
The truthful operator flow after implementation should be:

1. process a match via watcher or `soccer360 process`
2. find core processed outputs under `/tank/processed/<match>/`
3. find highlight delivery assets under `/tank/highlights/<match>/`
4. if hard frames were exported, run `scripts/labelstudio_import.sh <match>` to produce Label Studio tasks with pre-drawn rectangles when bbox data is available

Important truth constraints:
- the system is still heuristic and ball-first
- some matches may produce no highlights
- the delivery workflow is file-based, not a new dashboard packaging feature
- no parent-ready claims beyond the concrete files actually emitted

---

## 8. Implementation boundaries by module

### `scripts/labelstudio_import.sh`
Owns:
- reading `hard_frames.json`
- generating `labelstudio/tasks.json`
- bbox field compatibility handling

Should not own:
- changing pipeline manifests
- dataset building
- training orchestration

### `src/highlights.py`
Owns:
- event detection
- clip clustering/ranking
- clip export
- `highlights.json`
- `highlights_reel.mp4`
- no-clips / reel-failure highlight package behavior

Should not own:
- final move/copy into processed directories
- top-level match metadata authoring

### `src/exporter.py`
Owns:
- copying finalized highlight directory into `/tank/highlights/<match>/`
- writing match-level `metadata.json`
- referencing whether highlight outputs exist

Should not own:
- discovering clips after the fact
- generating reel internals
- recreating highlight manifest content

### `src/pipeline.py`
Owns:
- phase order only
- calling highlight generation before export finalization

Should not gain:
- inline reel/media logic
- special delivery-pack branch orchestration

---

## 9. Risks to avoid
- moving manifest compatibility fixes into multiple places instead of the import boundary
- pushing highlight package logic into exporter, which would blur responsibilities
- adding a new CLI command that duplicates `process`
- making docs promise a reel in cases where no clips exist
- inventing event/action semantics inside `highlights.json`

---

## 10. Recommended implementation sequence for Peter
1. **Label Studio compatibility repair**
2. **Highlight manifest generation**
3. **Reel generation + failure handling**
4. **Exporter metadata/reference cleanup**
5. **README / workflow doc truth-sync**

This order keeps the risk low and lets each step be verified independently.

---

## 11. Acceptance shape for this architecture
This milestone is architecturally complete when:
- Label Studio import accepts the current hard-frame manifest shape
- highlight generation writes `highlights.json` beside the clips
- highlight generation can assemble `highlights_reel.mp4`
- no-clips and reel-failure behavior are explicit and honest
- exporter references highlight outputs without becoming the highlight source of truth
- docs describe the file-based delivery workflow without inventing future product surfaces
