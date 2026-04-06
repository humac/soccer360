# Soccer360 SOTA Pipeline Upgrade Plan

## Objective
Upgrade the Soccer360 pipeline to generate human-operator quality broadcast videos from 180° panoramic cameras. This involves shifting ball detection from single-frame YOLO to a temporal heatmap architecture, upgrading player tracking to handle occlusions better, and implementing advanced cinematic camera controls.

## Background & Motivation
The current pipeline relies on single-frame YOLO models for both ball and player detection, smoothed by a Kalman filter and EMA in `src/camera.py`.
1. **Ball Detection Limits:** In high-resolution 180° video, the ball is tiny (<10 pixels) and heavily motion-blurred. Standard YOLO (which looks for edges in a single frame) drops detections frequently.
2. **Player Tracking Limits:** Crowded scenarios (corners, tackles) cause YOLO to mistakenly delete players via Non-Maximum Suppression (NMS), leading to ID switches in the ByteTrack tracker and jerky camera movements if Center of Play is enabled.
3. **Robotic Camera:** The current virtual camera follows the ball's movement too rigidly (using a velocity deadband). Human operators use a spatial dead-zone and anticipate passes.

## Proposed Solution

This plan is broken into three phased implementations to minimize disruption and allow incremental testing.

### Phase 1: Cinematic Camera Controller Upgrades (Immediate Impact)
These changes are mathematical and require zero new model training. They instantly make the output `broadcast.mp4` feel more natural.
* **Spatial Dead-Zone:** Modify `src/camera.py` to allow the ball to move freely within the center ~30% of the frame without triggering a camera pan. The camera only accelerates when the ball approaches the edge of this safe zone.
* **Lookahead Anticipation:** Utilize the Kalman filter's velocity state (`d_yaw`, `d_pitch`) to project the target slightly ahead of the ball's actual position, leading the action on fast passes.
* **Dynamic Center-of-Play Blending:** Instead of a static blend weight (`0.15`), dynamically shift focus based on ball velocity. High velocity = 95% ball focus. Low velocity (dribbling/stoppage) = 50% ball / 50% player cluster centroid focus.

### Phase 2: Ball Detection Architecture Shift (The SOTA Move)
* **Abandon YOLO for the Ball:** Stop using bounding-box models for the ball.
* **Implement Temporal Heatmaps:** Integrate a model like **TrackNet (v2/v3)** or **WASB**. These architectures take 3-5 consecutive frames as input and output a 2D probability map. They can detect the ball even when it is a motion-blurred streak.
* **Pipeline Integration:** Update `src/pipeline.py` and `src/detector.py` to buffer frames and pass temporal sequences to the new ball detection head.

### Phase 3: Player Detection & Tracking Upgrades
* **Replace ByteTrack with BoT-SORT:** Upgrade `src/tracker.py` to use BoT-SORT for player tracking. BoT-SORT integrates a Re-Identification (ReID) embedding model (like OSNet), drastically reducing ID switches when players cross paths.
* **Evaluate RT-DETR vs YOLO+SAHI:**
  * **RT-DETR:** Explore training an RT-DETR model for players. It treats detection as a direct set prediction, eliminating NMS heuristics, making it far superior in crowded penalty boxes.
  * **YOLO + SAHI:** If migrating away from YOLO is too costly, implement Slicing Aided Hyper Inference (SAHI) during inference. This slices the high-res 180° frame into patches before running YOLO, preserving resolution for tiny players on the far side of the pitch.

## Implementation Steps

1. **Step 1: Refactor `src/camera.py`**
   - Implement spatial dead-zone logic in `_clamp_pan_speed` or a dedicated cinematic framing method.
   - Implement lookahead anticipation using `kf.x` velocity data.
   - Add dynamic blending logic to `_tracks_to_angles_hybrid`.
2. **Step 2: Prototype Temporal Ball Detector**
   - Isolate a test script using a pre-trained TrackNet/WASB model on a Soccer360 video clip.
   - Compare the output trajectories against the current `yolo26l.pt` detections.
3. **Step 3: Integrate Dual-Path Detection in `src/pipeline.py`**
   - Fork the detection step: pass the current frame to YOLO (for players), and pass the frame buffer to the Temporal model (for the ball).
4. **Step 4: Upgrade Tracking Layer**
   - Import and integrate a BoT-SORT implementation into `src/tracker.py`.
   - Ensure the new tracker feeds stable cluster centroids to the upgraded camera controller.

## Verification & Rollback
* **Verification:** Run `pytest tests/test_camera.py` and visually inspect the generated `camera_path.json` on a known test match. The spatial dead-zone should result in longer periods of `d_yaw = 0` followed by smoother accelerations.
* **Rollback:** The Phase 1 changes in `src/camera.py` can be hidden behind configuration flags (e.g., `camera.cinematic_framing_enabled: true`) in `pipeline.yaml` to allow immediate fallback to the legacy velocity deadband. Phase 2 and 3 can be rolled out as a new `mode` in `pipeline.py`.
