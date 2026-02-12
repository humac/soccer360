#!/usr/bin/env bash
# Import hard frames for a match into Label Studio task format.
# Usage: scripts/labelstudio_import.sh <match_name>
#
# Reads frames from /tank/labeling/<match_name>/frames/ and creates a
# Label Studio task JSON at /tank/labeling/<match_name>/labelstudio/tasks.json.
# Predicted bboxes from hard_frames.json are included as pre-annotations.

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

# Generate Label Studio import JSON
python3 << 'PYEOF'
import json
import sys
from pathlib import Path

match_name = sys.argv[1]
frames_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])

# Load manifest for predicted bboxes
frame_meta = {}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    for f in manifest.get("frames", []):
        frame_meta[f["frame_index"]] = f

tasks = []
for img in sorted(frames_dir.glob("frame_*.jpg")):
    frame_idx = int(img.stem.split("_")[1])
    # Path as seen by Label Studio container
    # (mounted at /label-studio/data/labeling via docker-compose)
    ls_path = f"/data/labeling/{match_name}/frames/{img.name}"

    task = {
        "data": {
            "image": ls_path,
            "frame_index": frame_idx,
            "match_name": match_name,
        },
    }

    # Include predictions if we have a predicted bbox
    meta = frame_meta.get(frame_idx, {})
    if "predicted_bbox" in meta:
        bbox = meta["predicted_bbox"]
        # Label Studio expects percentages of image dimensions
        # Detection resolution is 1920x960 (from pipeline.yaml)
        img_w, img_h = 1920, 960
        x_pct = (bbox[0] / img_w) * 100
        y_pct = (bbox[1] / img_h) * 100
        w_pct = ((bbox[2] - bbox[0]) / img_w) * 100
        h_pct = ((bbox[3] - bbox[1]) / img_h) * 100

        task["predictions"] = [{
            "model_version": "ball_detector",
            "result": [{
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": round(x_pct, 2),
                    "y": round(y_pct, 2),
                    "width": round(w_pct, 2),
                    "height": round(h_pct, 2),
                    "rectanglelabels": ["ball"],
                },
            }],
        }]

    tasks.append(task)

output_file = output_dir / "tasks.json"
output_file.write_text(json.dumps(tasks, indent=2))
print(f"Created {len(tasks)} Label Studio tasks -> {output_file}")
PYEOF "$MATCH_NAME" "$FRAMES_DIR" "$LS_OUTPUT_DIR" "$MANIFEST"

echo ""
echo "To import into Label Studio:"
echo "  1. Open http://localhost:8080"
echo "  2. Create a new project (or select existing) for '${MATCH_NAME}'"
echo "  3. Settings -> Labeling Interface -> use 'Object Detection' template"
echo "     Add label: 'ball'"
echo "  4. Import tasks from: ${LS_OUTPUT_DIR}/tasks.json"
echo "  5. Label ball bounding boxes in each frame"
echo "  6. Export annotations in YOLO format to: ${LABELING_DIR}/labels/"
