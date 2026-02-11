#!/usr/bin/env bash
# Soccer360 Model Training Workflow
# Fine-tune YOLO ball detection model using labeled data.

set -euo pipefail

EPOCHS="${1:-50}"
DATA="${2:-/tank/labeling/dataset.yaml}"

echo "================================================"
echo "Soccer360 Model Training"
echo "  Epochs: $EPOCHS"
echo "  Data:   $DATA"
echo "================================================"

# Verify dataset exists
if [ ! -f "$DATA" ]; then
    echo "ERROR: Dataset YAML not found at $DATA"
    echo ""
    echo "To create a dataset:"
    echo "  1. Export hard frames:  docker compose run --rm worker soccer360 export-hard-frames <video> <detections.jsonl>"
    echo "  2. Label in Label Studio: http://localhost:8080"
    echo "  3. Export labels in YOLO format to /tank/labeling/"
    echo "  4. Create dataset.yaml pointing to train/val splits"
    exit 1
fi

echo ""
echo "Starting training..."
docker compose run --rm worker soccer360 train --epochs "$EPOCHS" --data "$DATA"

echo ""
echo "================================================"
echo "Training complete!"
echo "  New model saved to /tank/models/"
echo "  Latest best model: /tank/models/ball_best.pt"
echo ""
echo "To use the new model, update configs/pipeline.yaml:"
echo "  model.path: /tank/models/ball_best.pt"
echo "================================================"
