#!/usr/bin/env bash
# Soccer360 Model Training (Active Learning)
#
# Fine-tune YOLO ball detection model using labeled data.
# Uses timestamp-based versioning (YYYYMMDD_HHMM).
# Automatically promotes best model to /tank/models/ball_best.pt.
#
# Usage: scripts/train_ball.sh [epochs] [dataset_yaml]
#
# The worker will automatically pick up ball_best.pt on the next run.

set -euo pipefail

EPOCHS="${1:-50}"
DATA="${2:-/tank/labeling/dataset/dataset.yaml}"

VERSION=$(date +%Y%m%d_%H%M)
RUN_NAME="ball_model_${VERSION}"

echo "================================================"
echo "Soccer360 Model Training"
echo "  Run:    $RUN_NAME"
echo "  Epochs: $EPOCHS"
echo "  Data:   $DATA"
echo "================================================"

# Verify dataset exists
if [ ! -f "$DATA" ]; then
    echo "ERROR: Dataset YAML not found at $DATA"
    echo ""
    echo "To create a dataset:"
    echo "  1. Process videos:        docker compose up -d worker"
    echo "  2. Label in Label Studio: http://localhost:8080"
    echo "  3. Export labels (YOLO format) to /tank/labeling/<match>/labels/"
    echo "  4. Build dataset:         bash scripts/build_dataset.sh"
    exit 1
fi

echo ""
echo "Starting training on GPU 1..."
docker compose run --rm \
    -e SOCCER360_RUN_NAME="$RUN_NAME" \
    worker soccer360 train --epochs "$EPOCHS" --data "$DATA"

echo ""
echo "================================================"
echo "Training complete: $RUN_NAME"
echo "  Versioned model: /tank/models/$RUN_NAME/"
echo "  Best model:      /tank/models/ball_best.pt"
echo ""
echo "  The worker will automatically use ball_best.pt on the next run."
echo "  To reprocess a match:"
echo "    docker compose run --rm worker soccer360 process /tank/ingest/<match>.mp4"
echo "================================================"
