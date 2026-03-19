# Future Detection Settings Page Roadmap

## Summary
- Add a dedicated dashboard settings experience for video-processing parameters.
- Start from a readonly view of the effective config, then evolve into a controlled runtime-override system for future ingest jobs.
- Keep infrastructure and path settings in `configs/pipeline.yaml`; expose only runtime-safe processing knobs in the UI.

## Target End State
- A dedicated `Video Detection Settings` page in the dashboard.
- Runtime-safe settings can be edited from the UI and stored in a shared JSON override file under `/tank/data`.
- Future ingest jobs launched from the watcher or CLI `process` use:
  - base config from `configs/pipeline.yaml`
  - dashboard-managed runtime overrides
  - current ingest model selection
- Running jobs remain unchanged until the next ingest run.

## Planned Capabilities
- Group settings by subsystem:
  - `Detection`
  - `Filters / Ball Stabilization`
  - `Player Detection & Clustering`
  - `Camera / Auto-Follow`
  - `Reframer / Output`
  - `Highlights`
  - `Active Learning`
- Show current value, config path, help text, and whether a value comes from base config or runtime override.
- Provide presets for:
  - `Insta360 X5 8K 360`
  - `High-res 360 conservative`
  - `Dual action cam 4K flat (experimental)`
  - `Dual action cam 5.3K flat (experimental)`
- Highlight the most important knobs for a VEO-like auto-follow workflow:
  - model choice
  - confidence and image-size tradeoffs
  - player-cluster smoothing
  - ball-vs-cluster blend weight
  - pan speed, deadband, and FOV behavior
  - highlight scoring thresholds

## Runtime Override Design
- Use a dashboard-managed JSON file under `/tank/data` rather than editing `configs/pipeline.yaml`.
- Add helpers to:
  - load override state
  - validate supported keys and types
  - merge overrides onto the base config
  - clear/reset overrides
- Update watcher dispatch so each new job reads the latest effective config before creating a pipeline instance.
- Update CLI `process` to use the same merge logic.

## Constraints
- This does not add true multi-camera calibration, stitching, or flat-camera geometry support.
- Experimental flat-camera presets are tuning bundles for the current pipeline only.
- Unsafe infrastructure settings stay readonly and remain config-file managed.

## Suggested First Editable Version
- Detection thresholds and image size
- Player-cluster thresholds and smoothing
- Camera follow behavior and FOV limits
- Reframer output/FOV controls
- Highlight thresholds and score weights
- Active-learning export thresholds
