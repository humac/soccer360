"""Camera path generation: pixel coords -> yaw/pitch with Kalman smoothing.

Converts tracked ball positions in equirectangular space to smooth virtual
camera parameters (yaw, pitch, fov) suitable for broadcast-style viewing.

Pipeline:
  pixel (x,y) -> spherical (yaw, pitch) -> Kalman filter -> EMA smoothing
  -> pan speed clamping -> FOV computation -> camera_path.json
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from .utils import VideoMeta, load_json, pixel_to_yaw_pitch, write_json

logger = logging.getLogger("soccer360.camera")


def angle_diff(a: float, b: float) -> float:
    """Shortest signed angle from b to a, in degrees. Range: (-180, 180]."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def unwrap_angles(angles: list[float]) -> list[float]:
    """Unwrap a sequence of angles to remove 360-degree discontinuities."""
    if not angles:
        return []
    unwrapped = [angles[0]]
    for i in range(1, len(angles)):
        diff = angle_diff(angles[i], unwrapped[-1])
        unwrapped.append(unwrapped[-1] + diff)
    return unwrapped


def wrap_angle(a: float) -> float:
    """Wrap angle to [-180, 180) range."""
    return ((a + 180.0) % 360.0) - 180.0


class CameraPathGenerator:
    """Generate a smoothed camera path from tracked ball positions."""

    def __init__(self, config: dict):
        cam_cfg = config.get("camera", {})
        det_cfg = config.get("detector", {})
        v1_cfg = config.get("detection", {})

        self.max_pan_speed = cam_cfg.get("max_pan_speed_deg_per_sec", 60.0)
        self.max_fast_pan_speed = cam_cfg.get("max_fast_pan_speed_deg_per_sec", 120.0)
        self.ema_alpha = cam_cfg.get("ema_alpha", 0.15)
        self.default_fov = cam_cfg.get("default_fov", 90.0)
        self.min_fov = cam_cfg.get("min_fov", 80.0)
        self.max_fov = cam_cfg.get("max_fov", 100.0)
        self.lost_coast_frames = cam_cfg.get("lost_coast_frames", 30)
        self.lost_drift_frames = cam_cfg.get("lost_drift_frames", 90)
        self.field_center_yaw = cam_cfg.get("field_center_yaw_deg", 0.0)
        self.field_center_pitch = cam_cfg.get("field_center_pitch_deg", -5.0)

        # Deadband: suppress micro-oscillation
        self.deadband_deg = cam_cfg.get("deadband_deg", 0.5)
        self.velocity_threshold = cam_cfg.get("velocity_threshold_deg_per_sec", 2.0)
        self.lost_fov_widen = cam_cfg.get("lost_fov_widen", True)
        self.fov_ema_alpha = cam_cfg.get("fov_ema_alpha", 0.08)

        kalman_cfg = cam_cfg.get("kalman", {})
        self.process_noise = kalman_cfg.get("process_noise", 0.1)
        self.measurement_noise = kalman_cfg.get("measurement_noise", 1.0)

        if "detection" in config:
            img_size = v1_cfg.get("img_size", 960)
            self.det_width = img_size * 2
            self.det_height = img_size
        else:
            self.det_width = det_cfg.get("resolution", [1920, 960])[0]
            self.det_height = det_cfg.get("resolution", [1920, 960])[1]

        # Center of play config
        cop_cfg = config.get("center_of_play", {})
        self.cop_enabled = cop_cfg.get("enabled", False)
        self.cop_ball_blend = cop_cfg.get("ball_blend_weight", 0.15)
        self.cop_fov_from_spread = cop_cfg.get("fov_from_spread", True)
        self.cop_spread_max_fov = cop_cfg.get("spread_max_fov", 105.0)
        self.cop_spread_min_deg = cop_cfg.get("spread_min_deg", 15.0)
        self.cop_spread_max_deg = cop_cfg.get("spread_max_deg", 60.0)

    def generate_static(self, meta: VideoMeta, output_path: Path):
        """Generate a static camera path at field center for NO_DETECT mode."""
        camera_path = []
        for _ in range(meta.total_frames):
            camera_path.append({
                "yaw": self.field_center_yaw,
                "pitch": self.field_center_pitch,
                "fov": self.default_fov,
            })
        logger.info(
            "Static camera path: %d frames at yaw=%.1f, pitch=%.1f, fov=%.1f",
            len(camera_path), self.field_center_yaw,
            self.field_center_pitch, self.default_fov,
        )
        write_json(camera_path, output_path)

    def generate(
        self,
        tracks_path: Path,
        meta: VideoMeta,
        output_path: Path,
        player_cluster_path: Path | None = None,
    ):
        """Generate camera path from tracked ball positions.

        When player_cluster_path is provided and center_of_play is enabled,
        blends ball tracking with player cluster centroid for more robust
        camera following.
        """
        tracks = load_json(tracks_path)
        fps = meta.fps

        # Load player cluster data if available
        clusters = None
        if player_cluster_path is not None and self.cop_enabled:
            if player_cluster_path.exists():
                clusters = load_json(player_cluster_path)
                logger.info(
                    "Center of play enabled: loaded %d cluster entries", len(clusters)
                )

        logger.info(
            "Generating camera path: %d frames @ %.1f fps (hybrid=%s)",
            len(tracks), fps, clusters is not None,
        )

        # Step 1: Convert pixel coords to angles (hybrid or ball-only)
        if clusters is not None:
            raw_angles = self._tracks_to_angles_hybrid(tracks, clusters)
        else:
            raw_angles = self._tracks_to_angles(tracks)

        # Step 2: Kalman filter smoothing
        kalman_output = self._kalman_smooth(raw_angles, fps)

        # Step 3: EMA post-smoothing
        ema_output = self._ema_smooth(kalman_output)

        # Step 4: Pan speed clamping
        clamped = self._clamp_pan_speed(ema_output, fps)

        # Step 5: FOV computation (with optional spread data from clusters)
        if clusters is not None and self.cop_fov_from_spread:
            cluster_by_frame = {
                c["frame"]: c.get("cluster") for c in clusters
            }
            last_spread = None
            for i, entry in enumerate(clamped):
                cl = cluster_by_frame.get(i)
                if cl is not None:
                    last_spread = cl.get("spread_x_deg", 0.0)
                    entry["spread_x_deg"] = last_spread
                elif last_spread is not None:
                    entry["spread_x_deg"] = last_spread
        camera_path = self._compute_fov(clamped, fps)

        logger.info("Camera path generated: %d entries", len(camera_path))
        write_json(camera_path, output_path)

    def _tracks_to_angles(
        self, tracks: list[dict]
    ) -> list[tuple[float, float, float] | None]:
        """Convert per-frame ball pixel positions to (yaw, pitch, confidence)."""
        result = []
        for t in tracks:
            if t.get("ball") is not None:
                ball = t["ball"]
                yaw, pitch = pixel_to_yaw_pitch(
                    ball["x"], ball["y"],
                    self.det_width, self.det_height,
                )
                conf = ball.get("confidence", 1.0)
                result.append((yaw, pitch, conf))
            else:
                result.append(None)
        return result

    def _tracks_to_angles_hybrid(
        self, tracks: list[dict], clusters: list[dict]
    ) -> list[tuple[float, float, float] | None]:
        """Convert tracks + clusters to angles with priority blending.

        Priority: ball (high conf) > ball+cluster blend > cluster only > None.
        """
        cluster_by_frame = {c["frame"]: c.get("cluster") for c in clusters}
        result = []

        for i, t in enumerate(tracks):
            ball = t.get("ball")
            cluster = cluster_by_frame.get(i)

            if ball is not None and cluster is not None:
                # Blend ball position with cluster centroid
                ball_conf = ball.get("confidence", 0.5)
                blend = self.cop_ball_blend if ball_conf >= 0.5 else 0.5
                x = (1 - blend) * ball["x"] + blend * cluster["x"]
                y = (1 - blend) * ball["y"] + blend * cluster["y"]
                yaw, pitch = pixel_to_yaw_pitch(
                    x, y, self.det_width, self.det_height
                )
                result.append((yaw, pitch, ball_conf))
            elif ball is not None:
                # Ball only -- no cluster available
                yaw, pitch = pixel_to_yaw_pitch(
                    ball["x"], ball["y"], self.det_width, self.det_height
                )
                result.append((yaw, pitch, ball.get("confidence", 1.0)))
            elif cluster is not None:
                # Cluster only -- ball lost, follow player mass
                yaw, pitch = pixel_to_yaw_pitch(
                    cluster["x"], cluster["y"],
                    self.det_width, self.det_height,
                )
                result.append((yaw, pitch, 0.3))
            else:
                result.append(None)

        return result

    def _kalman_smooth(
        self,
        raw_angles: list[tuple[float, float, float] | None],
        fps: float,
    ) -> list[dict]:
        """Apply Kalman filter for smooth tracking with ball-lost prediction."""
        from filterpy.kalman import KalmanFilter

        dt = 1.0 / fps

        # State: [yaw, pitch, d_yaw, d_pitch]
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition (constant velocity)
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Process noise
        q = self.process_noise
        kf.Q = np.array([
            [q * dt**2, 0, q * dt, 0],
            [0, q * dt**2, 0, q * dt],
            [q * dt, 0, q, 0],
            [0, q * dt, 0, q],
        ])

        # Measurement noise
        r = self.measurement_noise
        kf.R = np.eye(2) * r

        # Initial covariance
        kf.P *= 100.0

        # Find first valid measurement to initialize
        init_yaw = self.field_center_yaw
        init_pitch = self.field_center_pitch
        for angle in raw_angles:
            if angle is not None:
                init_yaw, init_pitch = angle[0], angle[1]
                break

        kf.x = np.array([[init_yaw], [init_pitch], [0.0], [0.0]])

        output = []
        lost_count = 0
        prev_yaw = init_yaw

        for angle in raw_angles:
            kf.predict()

            if angle is not None:
                yaw, pitch, conf = angle

                # Handle yaw wrap-around: unwrap relative to filter state
                filter_yaw = float(kf.x[0, 0])
                yaw_unwrapped = filter_yaw + angle_diff(yaw, wrap_angle(filter_yaw))

                kf.update(np.array([[yaw_unwrapped], [pitch]]))
                lost_count = 0
            else:
                lost_count += 1

                # Decay velocity when lost for a while
                if lost_count > self.lost_coast_frames:
                    kf.x[2] *= 0.95  # d_yaw decay
                    kf.x[3] *= 0.95  # d_pitch decay

                # Drift toward field center when lost for extended period
                if lost_count > self.lost_drift_frames:
                    # Unwrap field center relative to current yaw
                    current_yaw = float(kf.x[0, 0])
                    target_yaw = current_yaw + angle_diff(
                        self.field_center_yaw, wrap_angle(current_yaw)
                    )
                    target_pitch = self.field_center_pitch
                    kf.x[0] += (target_yaw - kf.x[0]) * 0.02
                    kf.x[1] += (target_pitch - kf.x[1]) * 0.02

            output.append({
                "yaw": float(kf.x[0, 0]),
                "pitch": float(kf.x[1, 0]),
                "d_yaw": float(kf.x[2, 0]),
                "d_pitch": float(kf.x[3, 0]),
                "lost": angle is None,
                "lost_count": lost_count,
            })

        return output

    def _ema_smooth(self, kalman_output: list[dict]) -> list[dict]:
        """Apply exponential moving average post-smoothing."""
        alpha = self.ema_alpha
        smoothed = []

        if not kalman_output:
            return []

        prev_yaw = kalman_output[0]["yaw"]
        prev_pitch = kalman_output[0]["pitch"]

        for entry in kalman_output:
            yaw = prev_yaw + alpha * angle_diff(entry["yaw"], prev_yaw)
            pitch = prev_pitch + alpha * (entry["pitch"] - prev_pitch)

            prev_yaw = yaw
            prev_pitch = pitch

            smoothed.append({
                **entry,
                "yaw": yaw,
                "pitch": pitch,
            })

        return smoothed

    def _clamp_pan_speed(self, entries: list[dict], fps: float) -> list[dict]:
        """Enforce maximum angular velocity and deadband between consecutive frames.

        Deadband: ignore movements smaller than deadband_deg to prevent
        micro-oscillation. Velocity threshold: don't start moving camera
        until ball velocity exceeds minimum threshold.
        """
        if len(entries) <= 1:
            return entries

        max_delta_normal = self.max_pan_speed / fps  # deg per frame
        max_delta_fast = self.max_fast_pan_speed / fps
        deadband = self.deadband_deg
        vel_threshold = self.velocity_threshold / fps  # deg per frame

        clamped = [entries[0].copy()]

        for i in range(1, len(entries)):
            prev = clamped[-1]
            curr = entries[i].copy()

            # Smooth transition between normal and fast max speed
            ball_speed = math.sqrt(
                curr.get("d_yaw", 0) ** 2 + curr.get("d_pitch", 0) ** 2
            )
            speed_ratio = min(ball_speed / (max_delta_normal * 2), 1.0)
            max_delta = max_delta_normal + (max_delta_fast - max_delta_normal) * speed_ratio

            dyaw = angle_diff(curr["yaw"], prev["yaw"])
            dpitch = curr["pitch"] - prev["pitch"]

            # Deadband: suppress movements below threshold
            if abs(dyaw) < deadband:
                dyaw = 0.0
            if abs(dpitch) < deadband:
                dpitch = 0.0

            # Smooth gain reduction when ball speed is low
            if not curr.get("lost", False):
                gain_ratio = min(ball_speed / vel_threshold, 1.0)
                gain = 0.3 + 0.7 * gain_ratio
                dyaw *= gain
                dpitch *= gain

            # Clamp to max speed
            dyaw = np.clip(dyaw, -max_delta, max_delta)
            dpitch = np.clip(dpitch, -max_delta, max_delta)

            curr["yaw"] = prev["yaw"] + dyaw
            curr["pitch"] = prev["pitch"] + dpitch

            clamped.append(curr)

        return clamped

    def _compute_fov(self, entries: list[dict], fps: float) -> list[dict]:
        """Compute per-frame FOV based on ball velocity and lost state.

        Fast ball -> wider FOV (keep action in frame).
        Slow ball -> narrower FOV (tighter framing).
        Ball lost -> target max FOV (smoothed via EMA to avoid snap).

        All FOV values are EMA-smoothed to prevent frame-to-frame oscillation.
        """
        result = []
        fov_alpha = self.fov_ema_alpha
        prev_fov = self.default_fov

        for i, entry in enumerate(entries):
            speed = math.sqrt(
                entry.get("d_yaw", 0) ** 2 + entry.get("d_pitch", 0) ** 2
            )

            # Map speed to FOV range
            speed_normalized = min(speed / 5.0, 1.0)
            target_fov = self.min_fov + (self.max_fov - self.min_fov) * speed_normalized

            if entry.get("lost", False):
                if self.lost_fov_widen:
                    # Target max FOV when lost (EMA will smooth the transition)
                    target_fov = self.max_fov
                else:
                    lost_count = entry.get("lost_count", 0)
                    if lost_count > self.lost_coast_frames:
                        target_fov = self.default_fov

            # Widen FOV based on player spread when available
            spread = entry.get("spread_x_deg")
            if spread is not None and self.cop_fov_from_spread:
                t = (spread - self.cop_spread_min_deg) / (
                    self.cop_spread_max_deg - self.cop_spread_min_deg
                )
                t = max(0.0, min(1.0, t))
                spread_fov = self.min_fov + (
                    self.cop_spread_max_fov - self.min_fov
                ) * t
                target_fov = max(target_fov, spread_fov)

            # EMA smoothing: blend target with previous FOV
            if i == 0:
                fov = target_fov
            else:
                fov = prev_fov + fov_alpha * (target_fov - prev_fov)
            prev_fov = fov

            result.append({
                "yaw": wrap_angle(entry["yaw"]),
                "pitch": float(np.clip(entry["pitch"], -89.0, 89.0)),
                "fov": round(fov, 1),
            })

        return result
