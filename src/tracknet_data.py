"""TrackNetV3 dataset utilities: bbox-to-heatmap conversion + dataset loader.

Bridges the existing YOLO-format active learning labels to TrackNetV3's
heatmap ground truth format.  Also provides a PyTorch Dataset for loading
frame triplets with their corresponding heatmap targets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("soccer360.tracknet_data")


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------

def bbox_to_heatmap(
    cx: float,
    cy: float,
    img_h: int,
    img_w: int,
    sigma: float = 5.0,
) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered on (cx, cy).

    Args:
        cx, cy: ball center in pixel coordinates.
        img_h, img_w: heatmap dimensions.
        sigma: Gaussian standard deviation (pixels).

    Returns:
        (img_h, img_w) float32 array with values in [0, 1].
    """
    xs = np.arange(img_w, dtype=np.float32)
    ys = np.arange(img_h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def _yolo_label_to_ball_center(
    label_line: str,
    img_w: int,
    img_h: int,
    ball_class: int = 32,
) -> tuple[float, float, float, float] | None:
    """Parse a YOLO-format label line and return (cx, cy, w, h) in pixels.

    YOLO format: class cx cy w h (all normalized 0-1 except class).
    Returns None if the line is not for the ball class.
    """
    parts = label_line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    if cls != ball_class:
        return None
    cx = float(parts[1]) * img_w
    cy = float(parts[2]) * img_h
    w = float(parts[3]) * img_w
    h = float(parts[4]) * img_h
    return cx, cy, w, h


def convert_yolo_labels_to_heatmaps(
    labels_dir: Path,
    output_dir: Path,
    img_h: int = 288,
    img_w: int = 512,
    ball_class: int = 32,
    default_sigma: float = 5.0,
) -> int:
    """Convert YOLO-format label files to heatmap .npy files.

    For each .txt label file, finds the ball annotation and creates a
    Gaussian heatmap saved as .npy.  Sigma is proportional to bbox size
    when available.

    Returns:
        Number of heatmaps generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for label_path in sorted(labels_dir.glob("*.txt")):
        ball = None
        with open(label_path) as f:
            for line in f:
                result = _yolo_label_to_ball_center(
                    line, img_w, img_h, ball_class
                )
                if result is not None:
                    ball = result
                    break  # take first ball annotation

        if ball is None:
            # No ball in this frame — save zero heatmap (negative sample)
            heatmap = np.zeros((img_h, img_w), dtype=np.float32)
        else:
            cx, cy, w, h = ball
            sigma = max(default_sigma, max(w, h) * 0.5)
            heatmap = bbox_to_heatmap(cx, cy, img_h, img_w, sigma)

        out_path = output_dir / (label_path.stem + ".npy")
        np.save(out_path, heatmap)
        count += 1

    logger.info(
        "Converted %d labels to heatmaps in %s", count, output_dir
    )
    return count


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

def _load_tracknet_dataset_class():
    """Lazy-import the Dataset class to avoid requiring torch at module load."""
    import torch
    from torch.utils.data import Dataset

    class TrackNetDataset(Dataset):
        """Dataset for TrackNetV3 training: frame triplets + heatmap targets.

        Expects:
          frames_dir/  — sequential frame images (frame_000000.png, ...)
          heatmaps_dir/ — matching .npy heatmap files (frame_000000.npy, ...)

        Each sample returns:
          input: (9, H, W) float32 tensor — 3 consecutive RGB frames stacked
          target: (H, W) float32 tensor — center frame's ball heatmap
        """

        def __init__(
            self,
            frames_dir: Path,
            heatmaps_dir: Path,
            input_height: int = 288,
            input_width: int = 512,
        ):
            import cv2  # noqa: F811

            self.frames_dir = Path(frames_dir)
            self.heatmaps_dir = Path(heatmaps_dir)
            self.input_height = input_height
            self.input_width = input_width
            self._cv2 = cv2

            self.frame_paths = sorted(self.frames_dir.glob("*.png"))
            if not self.frame_paths:
                self.frame_paths = sorted(self.frames_dir.glob("*.jpg"))

            self.heatmap_paths = sorted(self.heatmaps_dir.glob("*.npy"))

            # Align by stem name
            heatmap_stems = {p.stem for p in self.heatmap_paths}
            self.valid_indices = [
                i
                for i, p in enumerate(self.frame_paths)
                if p.stem in heatmap_stems
            ]

        def __len__(self) -> int:
            return len(self.valid_indices)

        def _load_frame(self, idx: int) -> np.ndarray:
            """Load and resize a single frame."""
            path = self.frame_paths[idx]
            img = self._cv2.imread(str(path))
            if img is None:
                return np.zeros(
                    (self.input_height, self.input_width, 3), dtype=np.uint8
                )
            return self._cv2.resize(
                img, (self.input_width, self.input_height)
            )

        def __getitem__(self, index: int):
            center_idx = self.valid_indices[index]

            # Build triplet: [prev, center, next] with edge duplication
            prev_idx = max(0, center_idx - 1)
            next_idx = min(len(self.frame_paths) - 1, center_idx + 1)

            frames = []
            for idx in [prev_idx, center_idx, next_idx]:
                img = self._load_frame(idx)
                # HWC -> CHW, uint8 -> float32 [0, 1]
                t = (
                    torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                )
                frames.append(t)

            # Stack: 3 x (3, H, W) -> (9, H, W)
            input_tensor = torch.cat(frames, dim=0)

            # Load heatmap target
            heatmap_path = (
                self.heatmaps_dir / (self.frame_paths[center_idx].stem + ".npy")
            )
            heatmap = np.load(heatmap_path)
            target = torch.from_numpy(heatmap).float()

            return input_tensor, target

    return TrackNetDataset


def get_dataset_class():
    """Get the TrackNetDataset class (requires torch)."""
    return _load_tracknet_dataset_class()
