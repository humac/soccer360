#!/usr/bin/env bash
# Soccer360 Installation Script
# Run this on the server after completing SERVER_SETUP.md phases 1-7.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT="${PROJECT:-soccer360}"
IMAGE_TAG="${IMAGE_TAG:-soccer360-worker:local}"

echo "================================================"
echo "Soccer360 Installation"
echo "================================================"

# Verify prerequisites
echo "Checking prerequisites..."

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not installed. Complete SERVER_SETUP.md Phase 4 first."
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon not running or current user lacks permissions."
    echo "Try: sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: NVIDIA driver not installed. Complete SERVER_SETUP.md Phase 3 first."
    exit 1
fi

echo "  Docker:     $(docker --version)"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"

# Create directory structure
echo ""
echo "Creating directory structure..."

for dir in /tank/ingest /tank/processed /tank/highlights /tank/archive_raw /tank/models /tank/labeling /tank/logs /scratch/work; do
    if [ ! -d "$dir" ]; then
        sudo mkdir -p "$dir"
        sudo chown "$USER:$USER" "$dir"
        echo "  Created: $dir"
    else
        echo "  Exists:  $dir"
    fi
done

# Build Docker image
echo ""
echo "Building and verifying worker image (canonical verifier path)..."
cd "$REPO_DIR"
PROJECT="$PROJECT" IMAGE_TAG="$IMAGE_TAG" bash "$REPO_DIR/scripts/verify_container_assets.sh"

echo ""
echo "Pulling Label Studio image..."
docker compose -p "$PROJECT" pull labelstudio

# Verify
echo ""
echo "Verifying installation..."
docker compose -p "$PROJECT" run --rm worker soccer360 --help

echo ""
echo "================================================"
echo "Installation complete!"
echo ""
echo "Usage:"
echo "  Start watcher daemon:  docker compose -p $PROJECT up -d worker"
echo "  Start Label Studio:    docker compose -p $PROJECT up -d labelstudio"
echo "  Process single file:   docker compose -p $PROJECT run --rm worker soccer360 process /tank/ingest/match.mp4"
echo "  View logs:             docker compose -p $PROJECT logs -f worker"
echo "  Stop all:              docker compose -p $PROJECT down"
echo ""
echo "Drop 360 video files into /tank/ingest/ to begin processing."
echo "================================================"
