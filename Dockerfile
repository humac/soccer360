# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ULTRALYTICS_HOME=/app/.ultralytics
ENV YOLO_CONFIG_DIR=/app/.ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Dependencies layer: changes only when requirements-docker.txt changes.
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --upgrade pip
COPY requirements-docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-docker.txt

# Application source: changes often, but deps layer stays cached.
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY models/ models/

RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps .
RUN which soccer360

# Bake yolov8s.pt at a canonical path for V1 bootstrap detection
RUN mkdir -p /app/.ultralytics \
    && python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" \
    && if [ -s /app/yolov8s.pt ]; then MATCH="/app/yolov8s.pt"; else \
         MATCHES="$(find /app/.ultralytics -name 'yolov8s.pt' -type f -print)"; \
         COUNT="$(printf '%s\n' "$MATCHES" | grep -c .)"; \
         test "$COUNT" -eq 1; \
         MATCH="$MATCHES"; \
       fi \
    && if [ "$MATCH" != "/app/yolov8s.pt" ]; then cp "$MATCH" /app/yolov8s.pt; fi \
    && test -s /app/yolov8s.pt \
    && chown -R 1000:1000 /app/.ultralytics /app/yolov8s.pt \
    && echo "yolov8s.pt baked at /app/yolov8s.pt ($(stat -c%s /app/yolov8s.pt) bytes)" \
    || (echo "FATAL: expected exactly 1 yolov8s.pt under ULTRALYTICS_HOME, found: $MATCHES" && exit 1)

# Safety net: preserve runtime writability if future layers touch these paths.
RUN chown -R 1000:1000 /app/.ultralytics /app/yolov8s.pt || true

ENTRYPOINT ["soccer360"]
CMD ["watch"]
