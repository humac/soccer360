#!/usr/bin/env bash
# Import hard frames for a match into Label Studio task format.
# Usage: scripts/labelstudio_import.sh <match_name>
#
# Reads frames from /tank/labeling/<match_name>/frames/ and creates a
# Label Studio task JSON at /tank/labeling/<match_name>/labelstudio/tasks.json.
# Hard-frame manifests using either predicted_bbox/predicted_confidence or
# bbox/conf are supported for pre-annotations.

set -euo pipefail

MATCH_NAME="${1:?Usage: labelstudio_import.sh <match_name>}"
LABELING_DIR="/tank/labeling/${MATCH_NAME}"
FRAMES_DIR="${LABELING_DIR}/frames"
LS_OUTPUT_DIR="${LABELING_DIR}/labelstudio"
MANIFEST="${LABELING_DIR}/hard_frames.json"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "ERROR: Frames directory not found: $FRAMES_DIR"
    echo ""
    echo "Hard frames are exported automatically during pipeline processing."
    echo "Check that a match named '${MATCH_NAME}' has been processed."
    exit 1
fi

mkdir -p "$LS_OUTPUT_DIR"

python3 - "$MATCH_NAME" "$FRAMES_DIR" "$LS_OUTPUT_DIR" "$MANIFEST" <<'PYEOF'
import json
import sys
from pathlib import Path

from src.labelstudio_import import build_tasks

match_name = sys.argv[1]
frames_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])

tasks = build_tasks(match_name, frames_dir, manifest_path)
output_file = output_dir / "tasks.json"
output_file.write_text(json.dumps(tasks, indent=2))
print(f"Created {len(tasks)} Label Studio tasks -> {output_file}")
PYEOF

echo ""
echo "To import into Label Studio:"
echo "  1. Open http://localhost:8080"
echo "  2. Create a new project (or select existing) for '${MATCH_NAME}'"
echo "  3. Settings -> Labeling Interface -> use 'Object Detection' template"
echo "     Add label: 'ball'"
echo "  4. Import tasks from: ${LS_OUTPUT_DIR}/tasks.json"
echo "  5. Label ball bounding boxes in each frame"
echo "  6. Export annotations in YOLO format to: ${LABELING_DIR}/labels/"
