"""TrackNetV3 ball detection: temporal heatmap-based ball localization.

Takes 3 consecutive video frames as input and produces a 2D probability
heatmap.  The ball position is extracted via weighted centroid around the
peak.  This approach detects motion-blurred and sub-10px balls that
single-frame YOLO cannot see.

Architecture: lightweight encoder-decoder with 9-channel input (3 RGB
frames stacked) and 1-channel heatmap output.  The model definition is
vendored here to avoid an external dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("soccer360.tracknet")


# ---------------------------------------------------------------------------
# Model architecture (vendored TrackNetV3-style encoder-decoder)
# ---------------------------------------------------------------------------

def _build_tracknet_model(in_channels: int = 9):
    """Build a lightweight encoder-decoder for ball heatmap prediction.

    Input:  (B, 9, H, W)  — 3 consecutive RGB frames stacked
    Output: (B, 1, H, W)  — probability heatmap
    """
    import torch
    import torch.nn as nn

    class _ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class _UpBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            self.conv = _ConvBlock(out_ch * 2, out_ch)

        def forward(self, x, skip):
            x = self.up(x)
            # Handle size mismatch from odd dimensions
            dy = skip.size(2) - x.size(2)
            dx = skip.size(3) - x.size(3)
            if dy > 0 or dx > 0:
                x = nn.functional.pad(x, [0, dx, 0, dy])
            import torch as _t
            x = _t.cat([x, skip], dim=1)
            return self.conv(x)

    class TrackNetV3(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = _ConvBlock(in_channels, 64)
            self.enc2 = _ConvBlock(64, 128)
            self.enc3 = _ConvBlock(128, 256)
            self.pool = nn.MaxPool2d(2, 2)
            self.bottleneck = _ConvBlock(256, 512)
            self.up3 = _UpBlock(512, 256)
            self.up2 = _UpBlock(256, 128)
            self.up1 = _UpBlock(128, 64)
            self.out_conv = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.up3(b, e3)
            d2 = self.up2(d3, e2)
            d1 = self.up1(d2, e1)
            return torch.sigmoid(self.out_conv(d1))

    return TrackNetV3()


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

class TrackNetV3Detector:
    """Wraps TrackNetV3 inference: 3 frames in, (cx, cy, confidence) out."""

    def __init__(self, config: dict):
        tn_cfg = config.get("detection", {}).get("ball_model", {})
        self.model_path: str | None = tn_cfg.get("path")
        self.input_height: int = tn_cfg.get("input_height", 288)
        self.input_width: int = tn_cfg.get("input_width", 512)
        self.heatmap_threshold: float = tn_cfg.get("heatmap_threshold", 0.5)
        self.peak_radius: int = tn_cfg.get("peak_radius", 5)
        self.device: str = config.get("detection", {}).get("device", "cuda:0")
        self._model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load TrackNetV3 weights from disk."""
        import torch

        if self.model_path is None:
            raise RuntimeError("TrackNetV3 model path not configured")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"TrackNetV3 weights not found: {path}")

        model = _build_tracknet_model(in_channels=9)
        state = torch.load(str(path), map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self._model = model
        logger.info("TrackNetV3 loaded from %s on %s", path, self.device)

    def _ensure_model(self):
        if self._model is None:
            self._load_model()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frames: list[np.ndarray]) -> "torch.Tensor":
        """Resize, normalize, and stack 3 RGB frames into a 9-channel tensor."""
        import cv2
        import torch

        channels = []
        for frame in frames:
            resized = cv2.resize(
                frame, (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            # HWC -> CHW, uint8 -> float32 [0, 1]
            tensor = (
                torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
            )
            channels.append(tensor)

        # Stack: 3 x (3, H, W) -> (9, H, W)
        stacked = torch.cat(channels, dim=0)
        # Add batch dimension: (1, 9, H, W)
        return stacked.unsqueeze(0)

    # ------------------------------------------------------------------
    # Peak extraction
    # ------------------------------------------------------------------

    def _extract_peak(
        self, heatmap: np.ndarray
    ) -> tuple[float, float, float] | None:
        """Find ball position from heatmap via weighted centroid.

        Args:
            heatmap: 2D array (H, W) with values in [0, 1].

        Returns:
            (cx, cy, confidence) in heatmap pixel coords, or None.
        """
        peak_val = float(heatmap.max())
        if peak_val < self.heatmap_threshold:
            return None

        peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        h, w = heatmap.shape
        r = self.peak_radius

        # Extract neighborhood for sub-pixel refinement
        y1 = max(0, int(peak_y) - r)
        y2 = min(h, int(peak_y) + r + 1)
        x1 = max(0, int(peak_x) - r)
        x2 = min(w, int(peak_x) + r + 1)

        patch = heatmap[y1:y2, x1:x2]
        total = patch.sum()
        if total < 1e-8:
            return None

        # Weighted centroid within the patch
        ys, xs = np.mgrid[y1:y2, x1:x2]
        cx = float((xs * patch).sum() / total)
        cy = float((ys * patch).sum() / total)

        return cx, cy, peak_val

    # ------------------------------------------------------------------
    # Coordinate rescaling
    # ------------------------------------------------------------------

    def _rescale_coords(
        self, cx: float, cy: float, orig_w: int, orig_h: int
    ) -> tuple[float, float]:
        """Scale from model input resolution back to detection resolution."""
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height
        return cx * scale_x, cy * scale_y

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def predict(
        self,
        frames: list[np.ndarray],
        det_width: int | None = None,
        det_height: int | None = None,
    ) -> tuple[float, float, float] | None:
        """Run TrackNetV3 inference on a 3-frame window.

        Args:
            frames: exactly 3 RGB numpy arrays (H, W, 3).
            det_width: detection-space width for coordinate rescaling.
            det_height: detection-space height for coordinate rescaling.

        Returns:
            (cx, cy, confidence) in detection-space pixels, or None.
        """
        import torch

        self._ensure_model()

        if len(frames) < 3:
            # Pad with duplicates of the first frame
            while len(frames) < 3:
                frames = [frames[0]] + frames

        input_tensor = self._preprocess(frames[-3:]).to(self.device)

        with torch.no_grad():
            output = self._model(input_tensor)

        # (1, 1, H, W) -> (H, W)
        heatmap = output[0, 0].cpu().numpy()

        result = self._extract_peak(heatmap)
        if result is None:
            return None

        cx, cy, conf = result

        # Rescale to detection resolution if provided
        if det_width is not None and det_height is not None:
            cx, cy = self._rescale_coords(cx, cy, det_width, det_height)

        return cx, cy, conf
