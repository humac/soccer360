"""FastAPI monitoring dashboard for Soccer360 pipeline.

Serves a single-page HTML dashboard and provides a REST API + SSE
event stream for real-time pipeline monitoring and training management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .events import EventStore, create_event_store
from .metrics import gpu_utilization_snapshot

logger = logging.getLogger("soccer360.dashboard")

STATIC_DIR = Path(__file__).parent / "static"


def _scan_labeling_status(labeling_dir: Path) -> dict:
    """Scan labeling directory for matches with exported frames and labels."""
    matches = []
    total_frames = 0
    total_labeled = 0

    if not labeling_dir.is_dir():
        return {"matches": [], "total_frames": 0, "total_labeled": 0, "dataset_ready": False}

    for match_dir in sorted(labeling_dir.iterdir()):
        if not match_dir.is_dir() or match_dir.name in ("dataset",):
            continue

        frames_dir = match_dir / "frames"
        labels_dir = match_dir / "labels"

        frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
        label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

        if frame_count > 0:
            tasks_json = match_dir / "labelstudio" / "tasks.json"
            matches.append({
                "name": match_dir.name,
                "frames": frame_count,
                "labeled": label_count,
                "pct_labeled": round(label_count / frame_count * 100, 1) if frame_count > 0 else 0,
                "tasks_imported": tasks_json.exists(),
            })
            total_frames += frame_count
            total_labeled += label_count

    dataset_yaml = labeling_dir / "dataset" / "dataset.yaml"

    return {
        "matches": matches,
        "total_frames": total_frames,
        "total_labeled": total_labeled,
        "dataset_ready": dataset_yaml.exists(),
        "dataset_yaml": str(dataset_yaml) if dataset_yaml.exists() else None,
    }


def create_app(config: dict | None = None) -> FastAPI:
    """Create and configure the FastAPI dashboard application."""
    config = config or {}
    store = create_event_store(config)

    app = FastAPI(title="Soccer360 Dashboard", version="0.1.0")

    # CORS: allow Label Studio (port 8080) to fetch images from dashboard (port 8088)
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # Serve static files if directory exists
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # HTML entry point
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_file = STATIC_DIR / "index.html"
        if not index_file.exists():
            return HTMLResponse("<h1>Soccer360 Dashboard</h1><p>static/index.html not found</p>")
        return HTMLResponse(index_file.read_text())

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------

    @app.get("/api/status")
    async def status():
        return store.get_current_status()

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 50):
        return store.get_jobs(limit=limit)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        phases = store.get_phases(job_id)
        return {"job": job, "phases": phases}

    @app.get("/api/gpu")
    async def gpu_status():
        snap = gpu_utilization_snapshot()
        return snap or {"available": False}

    @app.get("/api/decisions/pending")
    async def pending_decisions():
        return store.get_pending_decisions()

    @app.post("/api/decisions/{decision_id}/resolve")
    async def resolve_decision(decision_id: int, request: Request):
        body = await request.json()
        response = body.get("response", "")
        decision_status = body.get("status", "approved")
        if decision_status not in ("approved", "rejected"):
            raise HTTPException(status_code=400, detail="status must be 'approved' or 'rejected'")

        decision = store.get_decision(decision_id)
        if not decision:
            raise HTTPException(status_code=404, detail="Decision not found")
        if decision["status"] != "pending":
            raise HTTPException(status_code=409, detail=f"Decision already resolved: {decision['status']}")

        store.resolve_decision(decision_id, response, status=decision_status)
        return {"ok": True, "decision_id": decision_id, "status": decision_status}

    # ------------------------------------------------------------------
    # Training / Active Learning
    # ------------------------------------------------------------------

    # In-memory training state (single-server, no persistence needed)
    _training_state = {"status": "idle", "log": [], "error": None}
    _training_lock = threading.Lock()

    labeling_dir = Path(config.get("paths", {}).get("labeling", "/tank/labeling"))
    models_dir = Path(config.get("paths", {}).get("models", "/tank/models"))

    @app.get("/api/training/labeling-status")
    async def labeling_status():
        return _scan_labeling_status(labeling_dir)

    @app.post("/api/training/import-tasks/{match_name}")
    async def import_tasks(match_name: str, request: Request):
        """Generate Label Studio import tasks for a match."""
        if ".." in match_name or "/" in match_name:
            raise HTTPException(status_code=400, detail="Invalid match name")
        match_dir = labeling_dir / match_name
        frames_dir = match_dir / "frames"
        if not frames_dir.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"No frames found for '{match_name}'. Process a video first.",
            )

        # Load manifest for predicted bboxes
        manifest_path = match_dir / "hard_frames.json"
        frame_meta = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                for f in manifest.get("frames", []):
                    frame_meta[f["frame_index"]] = f
            except Exception:
                pass

        # Build Label Studio task JSON
        # Use the Host header so URLs work from the browser's perspective
        host = request.headers.get("host", "localhost:8088")
        scheme = request.headers.get("x-forwarded-proto", "http")
        base_url = f"{scheme}://{host}"
        tasks = []
        img_w, img_h = 1920, 960  # detection resolution
        for img in sorted(frames_dir.glob("frame_*.jpg")):
            frame_idx = int(img.stem.split("_")[1])
            ls_path = f"{base_url}/api/labeling/frames/{match_name}/{img.name}"

            task: dict = {
                "data": {
                    "image": ls_path,
                    "frame_index": frame_idx,
                    "match_name": match_name,
                },
            }

            meta = frame_meta.get(frame_idx, {})
            bbox = meta.get("predicted_bbox") or meta.get("bbox")
            if bbox and len(bbox) == 4:
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

        # Write tasks.json
        ls_output_dir = match_dir / "labelstudio"
        ls_output_dir.mkdir(parents=True, exist_ok=True)
        tasks_file = ls_output_dir / "tasks.json"
        tasks_file.write_text(json.dumps(tasks, indent=2))

        return {"ok": True, "match": match_name, "tasks": len(tasks)}

    @app.get("/api/training/status")
    async def training_status():
        with _training_lock:
            return dict(_training_state)

    @app.get("/api/training/models")
    async def list_models():
        """List available models in /tank/models."""
        model_files = []
        if models_dir.is_dir():
            for f in sorted(models_dir.glob("**/*.pt")):
                model_files.append({
                    "path": str(f),
                    "name": f.name,
                    "size_mb": round(f.stat().st_size / 1e6, 1),
                    "is_active": f.name == "ball_best.pt",
                })
        return model_files

    @app.post("/api/training/build-dataset")
    async def build_dataset():
        """Trigger dataset build from labeled frames."""
        with _training_lock:
            if _training_state["status"] == "running":
                raise HTTPException(status_code=409, detail="Training or build already in progress")
            _training_state["status"] = "building"
            _training_state["log"] = []
            _training_state["error"] = None

        def _run_build():
            try:
                result = subprocess.run(
                    ["bash", "/app/scripts/build_dataset.sh", str(labeling_dir)],
                    capture_output=True, text=True, timeout=300,
                )
                with _training_lock:
                    _training_state["log"] = result.stdout.splitlines()
                    if result.returncode != 0:
                        _training_state["status"] = "failed"
                        _training_state["error"] = result.stderr or f"Exit code {result.returncode}"
                    else:
                        _training_state["status"] = "idle"
                logger.info("Dataset build completed (exit=%d)", result.returncode)
            except Exception as exc:
                with _training_lock:
                    _training_state["status"] = "failed"
                    _training_state["error"] = str(exc)
                logger.exception("Dataset build failed")

        threading.Thread(target=_run_build, daemon=True).start()
        return {"ok": True, "status": "building"}

    @app.post("/api/training/train")
    async def start_training(request: Request):
        """Trigger model training."""
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        epochs = body.get("epochs", 50)

        with _training_lock:
            if _training_state["status"] in ("running", "building"):
                raise HTTPException(status_code=409, detail="Training or build already in progress")
            _training_state["status"] = "running"
            _training_state["log"] = []
            _training_state["error"] = None

        dataset_yaml = labeling_dir / "dataset" / "dataset.yaml"
        if not dataset_yaml.exists():
            with _training_lock:
                _training_state["status"] = "failed"
                _training_state["error"] = "No dataset found. Build the dataset first."
            raise HTTPException(status_code=400, detail="Dataset not built. Run build-dataset first.")

        def _run_training():
            try:
                result = subprocess.run(
                    [
                        "bash", "/app/scripts/train_ball.sh",
                        str(epochs), str(dataset_yaml),
                    ],
                    capture_output=True, text=True, timeout=7200,  # 2hr max
                )
                with _training_lock:
                    _training_state["log"] = result.stdout.splitlines()[-50:]  # last 50 lines
                    if result.returncode != 0:
                        _training_state["status"] = "failed"
                        _training_state["error"] = result.stderr[-500:] if result.stderr else f"Exit code {result.returncode}"
                    else:
                        _training_state["status"] = "completed"
                logger.info("Training completed (exit=%d)", result.returncode)
            except subprocess.TimeoutExpired:
                with _training_lock:
                    _training_state["status"] = "failed"
                    _training_state["error"] = "Training timed out after 2 hours"
                logger.error("Training timed out")
            except Exception as exc:
                with _training_lock:
                    _training_state["status"] = "failed"
                    _training_state["error"] = str(exc)
                logger.exception("Training failed")

        threading.Thread(target=_run_training, daemon=True).start()
        return {"ok": True, "status": "running", "epochs": epochs}

    # ------------------------------------------------------------------
    # Labeling frame server (for Label Studio to fetch images)
    # ------------------------------------------------------------------

    @app.get("/api/labeling/frames/{match_name}/{filename}")
    async def serve_labeling_frame(match_name: str, filename: str):
        """Serve a hard frame image. Used by Label Studio tasks."""
        if ".." in match_name or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid path")
        frame_path = labeling_dir / match_name / "frames" / filename
        if not frame_path.is_file():
            raise HTTPException(status_code=404, detail="Frame not found")
        return FileResponse(str(frame_path), media_type="image/jpeg")

    # ------------------------------------------------------------------
    # Media / Processed Videos
    # ------------------------------------------------------------------

    processed_dir = Path(config.get("paths", {}).get("processed", "/tank/processed"))
    highlights_dir = Path(config.get("paths", {}).get("highlights", "/tank/highlights"))

    @app.get("/api/media/matches")
    async def list_matches():
        """List processed matches with available video files."""
        matches = []
        if processed_dir.is_dir():
            for match_dir in sorted(processed_dir.iterdir(), reverse=True):
                if not match_dir.is_dir() or match_dir.name.startswith("."):
                    continue
                videos = []
                for vf in sorted(match_dir.glob("*.mp4")):
                    videos.append({
                        "name": vf.name,
                        "size_mb": round(vf.stat().st_size / 1e6, 1),
                    })
                # Check for highlights
                hl_dir = highlights_dir / match_dir.name
                if hl_dir.is_dir():
                    for vf in sorted(hl_dir.glob("*.mp4")):
                        videos.append({
                            "name": f"highlights/{vf.name}",
                            "size_mb": round(vf.stat().st_size / 1e6, 1),
                        })
                if videos:
                    # Read metadata if available
                    meta_file = match_dir / "metadata.json"
                    meta = {}
                    if meta_file.exists():
                        try:
                            meta = json.loads(meta_file.read_text())
                        except Exception:
                            pass
                    matches.append({
                        "name": match_dir.name,
                        "videos": videos,
                        "mode": meta.get("mode", "--"),
                        "processed_at": meta.get("processing_start", "--"),
                    })
        return matches

    @app.get("/api/media/{match_name}/{filename:path}")
    async def stream_video(match_name: str, filename: str):
        """Serve a processed video file with range request support."""
        # Sanitize path components
        if ".." in match_name or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid path")

        if filename.startswith("highlights/"):
            clip_name = filename[len("highlights/"):]
            video_path = highlights_dir / match_name / clip_name
        else:
            video_path = processed_dir / match_name / filename

        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Video not found")

        return FileResponse(
            str(video_path),
            media_type="video/mp4",
            filename=filename,
        )

    # ------------------------------------------------------------------
    # SSE event stream
    # ------------------------------------------------------------------

    @app.get("/api/events")
    async def event_stream(request: Request, after: int = 0):
        """SSE endpoint that streams new phase events and periodic GPU snapshots."""

        async def generate():
            last_id = after
            gpu_counter = 0

            while True:
                if await request.is_disconnected():
                    break

                # Stream new phase events
                events = store.get_events_since(after_id=last_id)
                for ev in events:
                    last_id = ev["id"]
                    yield {
                        "event": f"phase_{ev['status']}",
                        "id": str(last_id),
                        "data": json.dumps(ev),
                    }

                # Stream pending decisions
                decisions = store.get_pending_decisions()
                for d in decisions:
                    yield {
                        "event": "decision_pending",
                        "data": json.dumps(d),
                    }

                # Periodic GPU snapshot (every ~5 seconds = 5 iterations)
                gpu_counter += 1
                if gpu_counter >= 5:
                    gpu_counter = 0
                    snap = gpu_utilization_snapshot()
                    if snap:
                        yield {
                            "event": "gpu_snapshot",
                            "data": json.dumps(snap),
                        }

                # Periodic status heartbeat
                status = store.get_current_status()
                yield {
                    "event": "status",
                    "data": json.dumps(status),
                }

                await asyncio.sleep(1)

        return EventSourceResponse(generate())

    return app
