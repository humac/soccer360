#!/usr/bin/env python3
"""Train TrackNetV3 on labeled ball data exported by active learning.

Usage:
    python scripts/train_tracknet.py \
        --frames /tank/labeling/match_001/images \
        --labels /tank/labeling/match_001/labels \
        --output /tank/models/tracknet_v1 \
        --epochs 100

The script:
1. Converts YOLO-format labels to Gaussian heatmaps
2. Builds a TrackNetDataset (frame triplets + heatmaps)
3. Trains the vendored TrackNetV3 architecture with weighted focal loss
4. Saves versioned checkpoints and a best model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger("soccer360.train_tracknet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TrackNetV3 ball detector"
    )
    parser.add_argument(
        "--frames", required=True, type=Path,
        help="Directory of sequential frame images (.png or .jpg)",
    )
    parser.add_argument(
        "--labels", required=True, type=Path,
        help="Directory of YOLO-format label files (.txt)",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output directory for model checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--base-model", type=Path, default=None,
        help="Pre-trained TrackNetV3 weights to fine-tune from",
    )
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--ball-class", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device for training (cuda:0, cpu, etc.)",
    )
    return parser.parse_args()


def weighted_focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for imbalanced heatmap prediction (ball pixels are rare)."""
    import torch

    bce = torch.nn.functional.binary_cross_entropy(
        pred, target, reduction="none"
    )
    pt = torch.where(target > 0.5, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def train(args: argparse.Namespace):
    import torch
    from torch.utils.data import DataLoader, random_split

    from src.tracknet import _build_tracknet_model
    from src.tracknet_data import convert_yolo_labels_to_heatmaps, get_dataset_class

    # Step 1: Convert labels to heatmaps
    heatmaps_dir = args.output / "heatmaps"
    logger.info("Converting YOLO labels to heatmaps...")
    count = convert_yolo_labels_to_heatmaps(
        labels_dir=args.labels,
        output_dir=heatmaps_dir,
        img_h=args.input_height,
        img_w=args.input_width,
        ball_class=args.ball_class,
    )
    logger.info("Generated %d heatmaps", count)

    if count == 0:
        logger.error("No labels found — nothing to train on")
        sys.exit(1)

    # Step 2: Build dataset
    TrackNetDataset = get_dataset_class()
    dataset = TrackNetDataset(
        frames_dir=args.frames,
        heatmaps_dir=heatmaps_dir,
        input_height=args.input_height,
        input_width=args.input_width,
    )
    logger.info("Dataset: %d samples", len(dataset))

    if len(dataset) == 0:
        logger.error("No matching frame/heatmap pairs found")
        sys.exit(1)

    # Train/val split
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
    )

    # Step 3: Build model
    model = _build_tracknet_model(in_channels=9)
    if args.base_model is not None:
        state = torch.load(str(args.base_model), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded base model from %s", args.base_model)

    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5,
    )

    # Step 4: Training loop
    args.output.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            pred = model(inputs).squeeze(1)  # (B, H, W)
            loss = weighted_focal_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                pred = model(inputs).squeeze(1)
                val_loss += weighted_focal_loss(pred, targets).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d  train_loss=%.5f  val_loss=%.5f  lr=%.2e",
            epoch, args.epochs, train_loss, val_loss, lr,
        )

        # Save checkpoint
        if epoch % 10 == 0:
            ckpt_path = args.output / f"tracknet_epoch{epoch:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = args.output / "tracknet_best.pt"
            torch.save(model.state_dict(), best_path)
            logger.info("New best model saved (val_loss=%.5f)", val_loss)

    logger.info("Training complete. Best val_loss=%.5f", best_val_loss)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
