# Future Features Backlog

## Overview
- Purpose: maintain a practical roadmap for reliability, active learning, quality/performance, and operations.
- Scope: future work only; no immediate implementation.
- Environment assumptions:
- Paths: `/tank/ingest`, `/scratch/work`, `/tank/processed`, `/tank/highlights`, `/tank/models`, `/tank/labeling`, `/tank/logs`, `/tank/archive_raw`, `/backup`
- Primary GPU: device `1` (Tesla P40) for inference/training.

## Priorities (P0/P1/P2)
- `P0 Must-have reliability/ops`: items 1-5
- `P1 Active learning + highlight quality`: items 6-12
- `P2 Performance/ergonomics/monitoring`: items 13-20

## Feature List

### P0 — Must-have reliability/ops

#### P0-01 Safe ingest / atomic copy
- Problem: partial copies trigger ffprobe/read failures and unstable jobs.
- Proposed Solution: watcher ignores `*.part`, `*.tmp`, hidden files; only processes allowed extensions (`.mp4`, `.mov`) after size stability for N seconds; document "copy as `.part` then rename".
- Config/Paths Impact: `watcher.ignore_suffixes`, stability window config; ingest at `/tank/ingest`.
- Acceptance Criteria:
- Files ending `.part`, `.tmp`, hidden are ignored.
- File is only queued after size unchanged for configured interval.
- README/runbook includes `.part` rename workflow.
- Effort: `S`
- Risks: very slow network copies may appear stable briefly if polling is too coarse.

#### P0-02 Model store + fallback
- Problem: missing model can crash pipeline.
- Proposed Solution: resolve model in order `/tank/models/ball_best.pt` -> `/app/models/ball_base.pt` -> `NO_DETECT`; log chosen mode/path.
- Config/Paths Impact: model path resolution, `/tank/models`, `/app/models`.
- Acceptance Criteria:
- Logs always show model source and path/mode.
- Missing `/tank/models/ball_best.pt` does not crash.
- Missing both models switches to `NO_DETECT`.
- Effort: `S`
- Risks: ambiguity if stale symlink in `/tank/models`.

#### P0-03 NO_DETECT mode outputs
- Problem: current failure path can prevent usable outputs when detection unavailable.
- Proposed Solution: generate `tactical_wide.mp4` and static-path `broadcast.mp4`; write `metadata.json` with `mode="no_detect"`.
- Config/Paths Impact: output flow under `/tank/processed/<match>`.
- Acceptance Criteria:
- Pipeline completes without detector model.
- Both videos exist for no-detect jobs.
- Metadata records `mode=no_detect` and model source.
- Effort: `M`
- Risks: broadcast quality reduced by static framing.

#### P0-04 Container model mount standardization
- Problem: repo-local `./models` diverges from persistent model store.
- Proposed Solution: standardize compose mount to `/tank/models:/app/models`; verify write permissions.
- Config/Paths Impact: `docker-compose.yml`, `/tank/models`.
- Acceptance Criteria:
- Worker sees `/app/models/ball_best.pt` when present in `/tank/models`.
- Training output persists across container recreates.
- Permission guidance documented.
- Effort: `S`
- Risks: host ownership mismatch causes permission denied.

#### P0-05 Ownership/permissions hardening
- Problem: root-owned paths break automation and reproducibility.
- Proposed Solution: define expected owner/group/mode for `/tank/*` and repo path ownership; add setup/verification notes.
- Config/Paths Impact: `/tank` datasets and `/tank/pipeline` repo ownership docs.
- Acceptance Criteria:
- Documented `chown/chmod` baseline and verification commands.
- Watcher/training run without sudo in normal ops.
- Effort: `S`
- Risks: mixed-user admin practices may reintroduce root-owned artifacts.

### P1 — Active learning loop (makes system improve)

#### P1-06 Hard-frame exporter during processing
- Problem: labeling data collection is manual and inconsistent.
- Proposed Solution: auto-export hard frames for low confidence, lost-ball streaks, large track jumps into `/tank/labeling/<match>/frames`; write manifest.
- Config/Paths Impact: `/tank/labeling/<match>/frames`, `/tank/labeling/<match>/hard_frames.json`, thresholds in config.
- Acceptance Criteria:
- Manifest contains frame index, timestamp, reason, optional bbox/conf.
- Exported frames exist for each manifest entry.
- Trigger reasons include low-confidence, loss, jump.
- Effort: `M`
- Risks: over-export volume if thresholds too permissive.

#### P1-07 Label Studio workflow hardening
- Problem: imports/tasks are manual and error-prone.
- Proposed Solution: helper script generates task JSON under `/tank/labeling/<match>/labelstudio`; docs constrain labeling to ball bbox class only.
- Config/Paths Impact: labelstudio import artifacts and README workflow.
- Acceptance Criteria:
- `scripts/labelstudio_import.sh <match>` produces valid import JSON.
- Tasks resolve local frame files in Label Studio.
- Documentation includes exact URL and steps.
- Effort: `S`
- Risks: Label Studio local-files setup differs by deployment.

#### P1-08 Dataset build automation
- Problem: assembling YOLO train/val data is ad hoc.
- Proposed Solution: `scripts/build_dataset.sh` aggregates labeled frames into YOLO structure and writes `/tank/labeling/dataset.yaml`.
- Config/Paths Impact: `/tank/labeling/train|val/images|labels`, `/tank/labeling/dataset.yaml`.
- Acceptance Criteria:
- Script builds deterministic train/val split.
- `dataset.yaml` points to generated paths.
- Handles missing/invalid label files with warnings.
- Effort: `M`
- Risks: inconsistent label formats across exports.

#### P1-09 Training + model promotion
- Problem: model refresh process is inconsistent and hard to audit.
- Proposed Solution: `scripts/train_ball.sh [epochs]` trains on GPU 1, writes `/tank/models/ball_vTIMESTAMP.pt`, promotes best to `/tank/models/ball_best.pt`, logs to `/tank/logs`.
- Config/Paths Impact: model artifacts and training logs.
- Acceptance Criteria:
- Training command pins to GPU device 1.
- Versioned model file created per run.
- `ball_best.pt` updated by copy/symlink promotion rule.
- Logs written under `/tank/logs`.
- Effort: `M`
- Risks: accidental promotion of regressed model without eval gate.

#### P1-10 Reprocess old matches with new model
- Problem: historical outputs do not benefit from model improvements.
- Proposed Solution: add reprocess command/script for prior ingest sources; support overwrite or create new output folder.
- Config/Paths Impact: `/tank/processed/<match>`, possible `_runN` behavior.
- Acceptance Criteria:
- Operator can target old file and run with latest model.
- Clear mode for overwrite vs new-folder output.
- Reprocessing action logged.
- Effort: `M`
- Risks: accidental overwrite of trusted outputs.

### P1 — Highlight detection upgrades

#### P1-11 Highlight heuristics upgrades
- Problem: false positives from midfield action.
- Proposed Solution: combine goal-box proximity + speed spikes + camera pan/zoom magnitude; add pre/post padding config, dedup logic, max clips per match.
- Config/Paths Impact: highlight thresholds and clip limits in config.
- Acceptance Criteria:
- Fewer midfield false highlights on validation set.
- More clips around shots/corners/scrambles.
- Max clips and dedup behavior are configurable.
- Effort: `L`
- Risks: overfitting heuristics to specific field/camera placement.

#### P1-12 Optional NO_DETECT highlights via motion proxy
- Problem: no ball track means highlights may be missing entirely.
- Proposed Solution: optional fallback using camera/global motion proxies; output includes disclaimer "no ball track".
- Config/Paths Impact: highlight mode flag and metadata annotation.
- Acceptance Criteria:
- In no-detect mode, optional proxy highlights can be produced.
- Clips and metadata include clear disclaimer.
- Effort: `M`
- Risks: high false-positive rate from non-game camera motion.

### P2 — Performance & quality

#### P2-13 TensorRT export support
- Problem: inference throughput may be suboptimal on P40.
- Proposed Solution: optional `.engine` export and auto-select if `tensorrt_path` exists; benchmark FPS.
- Config/Paths Impact: model backend/tensorrt path entries, `/tank/models`.
- Acceptance Criteria:
- Engine can be generated and loaded.
- Documented FPS comparison vs fp32 baseline.
- Safe fallback to `.pt` if engine unavailable.
- Effort: `L`
- Risks: engine compatibility drift across CUDA/container changes.

#### P2-14 Batch/segment processing & resume
- Problem: crashes force full reruns and wasted compute.
- Proposed Solution: resume from partial `/scratch/work/<job>` and skip completed phases.
- Config/Paths Impact: scratch work state files/checkpoints.
- Acceptance Criteria:
- Restart resumes completed phases without redoing them.
- Final outputs equivalent to clean run.
- Effort: `L`
- Risks: stale partial artifacts causing inconsistent outputs.

#### P2-15 Better scratch cleanup policy
- Problem: either over-cleaning (lose debug context) or disk bloat.
- Proposed Solution: retention-days policy, debug no-delete flag, periodic cleanup notes/cron.
- Config/Paths Impact: `/scratch/work` retention config and ops docs.
- Acceptance Criteria:
- Cleanup respects retention and debug-protect mode.
- Documented periodic cleanup process.
- Effort: `S`
- Risks: misconfigured retention fills NVMe.

#### P2-16 Multi-GPU selection
- Problem: hardcoded GPU selection limits flexibility.
- Proposed Solution: config-driven GPU index selection, default P40 (`1`), optional P2000 use for parallel workloads.
- Config/Paths Impact: GPU index setting in config/compose.
- Acceptance Criteria:
- Inference/training obey configured GPU index.
- Default remains GPU 1.
- Effort: `M`
- Risks: incorrect device mapping in container runtime.

### P2 — Insta360 ingestion ergonomics

#### P2-17 Insta360 input prep guide
- Problem: raw Insta360 `.insv` recordings are not directly processable.
- Proposed Solution: document X5 two-file recording behavior and required stitch/export to equirectangular MP4; optional validation helper for expected dimensions (e.g., 5760x2880).
- Config/Paths Impact: ingest SOP docs, optional preflight script.
- Acceptance Criteria:
- Guide clearly states stitch/export prerequisite.
- Operator can validate input format before ingest.
- Effort: `S`
- Risks: model assumptions break on non-equirectangular exports.

#### P2-18 Optional automatic stitch trigger (idea)
- Problem: manual stitch step is operational friction.
- Proposed Solution: documented future concept to detect `.insv` pairs in a staging folder and call external stitch tool.
- Config/Paths Impact: staging path and external-tool integration notes.
- Acceptance Criteria:
- Documented design only (no implementation).
- Includes guardrails and failure-handling considerations.
- Effort: `L`
- Risks: brittle dependency on external proprietary tooling.

### P2 — Ops / monitoring

#### P2-19 Health checks
- Problem: degraded services may go unnoticed; logs may grow unchecked.
- Proposed Solution: Docker healthchecks for worker "watch mode alive" and Label Studio; document log rotation strategy under `/tank/logs`.
- Config/Paths Impact: compose healthchecks, log retention/rotation docs.
- Acceptance Criteria:
- Healthcheck status visible in `docker compose ps`.
- Log rotation policy documented and testable.
- Effort: `M`
- Risks: false unhealthy states if checks are too strict.

#### P2-20 Metrics
- Problem: limited observability for throughput/performance.
- Proposed Solution: write per-job stats JSON (fps processed, GPU util snapshot, total time) to `/tank/processed/<match>/metadata.json`.
- Config/Paths Impact: metadata schema extension in processed outputs.
- Acceptance Criteria:
- Metadata includes timing and throughput fields.
- GPU snapshot present when available, null-safe otherwise.
- Effort: `M`
- Risks: GPU metric collection may vary by driver/container permissions.

## Notes/Links
- Home server assumptions: Docker Compose, ZFS datasets, Tesla P40 on GPU device 1.
- Operational paths reference:
- `/tank/ingest`, `/scratch/work`, `/tank/processed`, `/tank/highlights`, `/tank/models`, `/tank/labeling`, `/tank/logs`, `/tank/archive_raw`, `/backup`
- Cross-reference README operational sections for ingest, labeling, training, and reprocessing once implemented.

## Known Issues Encountered (Appendix)
- Docker moved to `/tank/docker` caused missing NVIDIA runtime until `daemon.json` updated.
- Initial ffprobe failures due to partial MP4 copy (`moov atom not found`).
- Compose command double-called `soccer360` because Dockerfile `ENTRYPOINT` already set.
