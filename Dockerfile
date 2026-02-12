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

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY models/ models/

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir .
RUN which soccer360

# Bake yolov8s.pt at a canonical path for V1 bootstrap detection
RUN mkdir -p /app/.ultralytics \
    && python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" \
    && MATCHES="$(find /app/.ultralytics -name 'yolov8s.pt' -type f -print)" \
    && COUNT="$(printf '%s\n' "$MATCHES" | grep -c .)" \
    && test "$COUNT" -eq 1 \
    && cp "$MATCHES" /app/yolov8s.pt \
    && test -s /app/yolov8s.pt \
    && echo "yolov8s.pt baked at /app/yolov8s.pt ($(stat -c%s /app/yolov8s.pt) bytes)" \
    || (echo "FATAL: expected exactly 1 yolov8s.pt under ULTRALYTICS_HOME, found: $MATCHES" && exit 1)

ENTRYPOINT ["soccer360"]
CMD ["watch"]
