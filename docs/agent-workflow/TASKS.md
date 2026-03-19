# TASKS — Soccer360

## Milestone
**Ball-First Delivery Pack**

## Goal
Sharpen the existing ball-first pipeline so a normal successful run can produce:
- Label Studio-ready hard-frame imports with pre-annotations when bbox data exists
- `highlights.json`
- `highlights_reel.mp4`
- a truthful documented operator delivery flow

## Definition of done
- current hard-frame manifests import into Label Studio with expected pre-drawn rectangles
- highlight outputs include a canonical manifest beside the clips
- a reel is generated from exported clips when clips exist
- no-clips and reel-failure behavior are defined and implemented truthfully
- docs describe only what the code actually emits

## Non-goals
Do **not** include any of the following in this milestone:
- player detection/tracking expansion
- action/event intelligence expansion
- new dashboard UX
- parent portal work
- overlays / annotated renders
- new `soccer360 pack` command

---

## Peter packet order

### Packet 1 — Lock down Label Studio manifest compatibility
**Files likely touched:**
- `scripts/labelstudio_import.sh`
- `tests/...` for import compatibility coverage

#### Tasks
1. Verify the exact hard-frame manifest shapes currently produced in normal code paths.
2. Update Label Studio import logic to accept both:
   - `predicted_bbox` / `predicted_confidence`
   - `bbox` / `conf`
3. Keep task generation deterministic and backward-compatible.
4. Add focused tests for both manifest shapes.

#### Acceptance
- current active-learning manifest shape produces Label Studio tasks with pre-drawn rectangles
- legacy-compatible manifest shape still works
- tests assert rectangle prediction payload exists when bbox data exists

#### Notes
- fix this at the import/helper boundary, not by broad pipeline manifest rewrites
- if bbox data is absent, task creation should still succeed without fake predictions

---

### Packet 2 — Make `highlights.json` a real canonical artifact
**Files likely touched:**
- `src/highlights.py`
- `tests/test_highlights.py`

#### Tasks
1. Normalize the highlight manifest shape around current real data.
2. Ensure `highlights.json` is written only when clips are actually exported.
3. Include only bounded fields:
   - filename
   - rank
   - score
   - start/end/duration
   - event types
   - event count
   - detector stats
4. Decide and implement how reel presence is represented in the manifest.

#### Acceptance
- processed matches with highlight clips produce `highlights.json`
- manifest ordering is deterministic
- manifest does not claim player/action semantics the system does not know
- tests cover schema/content for a multi-clip case

#### Notes
- `highlights.json` is the source of truth for highlight package metadata
- keep it beside the clips, not under processed output as the canonical copy

---

### Packet 3 — Build reel assembly inside the highlight subsystem
**Files likely touched:**
- `src/highlights.py`
- `tests/test_highlights.py`

#### Tasks
1. Add reel assembly after clip export, still inside highlight generation.
2. Use deterministic clip ordering based on the actual exported clip list.
3. Generate `highlights_reel.mp4` when one or more clips exist.
4. Handle these cases explicitly:
   - no clips -> skip reel cleanly
   - clips exist + concat succeeds -> reel written
   - clips exist + concat fails -> clips and manifest preserved, reel absent, failure surfaced
5. Add focused tests around concat input generation / order / no-clips behavior.

#### Acceptance
- `highlights_reel.mp4` appears beside clips when clips exist
- reel order is deterministic
- no-clips behavior does not create phantom files
- reel failure does not silently pretend success

#### Notes
- do not move this into `src/exporter.py`
- keep exporter as finalization, not media-editing logic

---

### Packet 4 — Reference highlight package state in match metadata
**Files likely touched:**
- `src/exporter.py`
- exporter tests

#### Tasks
1. Add minimal highlight package references to `metadata.json`.
2. Keep match metadata high-level; do not duplicate full clip manifest there.
3. Reflect truthful status, for example:
   - no highlights
   - clips only
   - clips + reel
   - reel failed
4. Ensure exporter continues copying the highlight directory as-is.

#### Acceptance
- operators can tell from `metadata.json` whether a highlight package exists
- no duplicate detailed clip schema is introduced in match metadata
- exporter remains free of highlight discovery/rebuild logic

#### Notes
- if current exporter tests are narrow, add just enough coverage to protect the status/reference behavior

---

### Packet 5 — Truthful operator workflow docs
**Files likely touched:**
- `README.md`
- possibly milestone workflow docs if needed

#### Tasks
1. Update docs so the standard operator flow matches actual implementation.
2. Explicitly state where to find:
   - processed outputs
   - highlight package outputs
   - Label Studio tasks
3. Document the no-clips case honestly.
4. Remove or avoid any wording that implies future parent/action features.

#### Acceptance
- docs tell an operator exactly where outputs land
- docs do not promise highlight assets when none are produced
- docs do not imply new dashboard/parent product behavior

---

## Cross-packet constraints
- Keep packets small and sequential.
- Avoid touching unrelated pipeline phases.
- Do not expand the clip scoring model beyond what is already in place.
- Do not turn this into a generalized packaging framework.

---

## Suggested verification targets per packet

### Packet 1
- importer unit test for `bbox`
- importer unit test for `predicted_bbox`
- manual sample task JSON review

### Packet 2
- highlight manifest unit test
- deterministic filename/order assertion

### Packet 3
- concat input/order unit test
- no-clips unit test
- reel-failure behavior test via mocked ffmpeg failure

### Packet 4
- exporter metadata test for highlight status/reference

### Packet 5
- docs readback verification against actual output paths

---

## Handoff note for Peter
If implementation pressure forces tradeoffs, preserve this priority order:
1. truthful Label Studio import compatibility
2. truthful `highlights.json`
3. truthful reel generation behavior
4. truthful metadata references
5. truthful docs

The rule is simple: **better a smaller honest delivery pack than a broader misleading one.**
