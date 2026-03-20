"""Tests for the FastAPI dashboard application."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch
import zipfile

import anyio
import httpx
import pytest

from src.dashboard import _build_dataset_from_labels, create_app
from src.events import EventStore
from src.watcher import ProcessedIngestStore


@pytest.fixture
def store():
    return EventStore(":memory:")


@pytest.fixture
def dashboard_config(tmp_path: Path):
    paths = {
        "ingest": str(tmp_path / "ingest"),
        "processed": str(tmp_path / "processed"),
        "highlights": str(tmp_path / "highlights"),
        "labeling": str(tmp_path / "labeling"),
        "models": str(tmp_path / "models"),
        "stagging": str(tmp_path / "stagging"),
        "archive_raw": str(tmp_path / "archive_raw"),
    }
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    (Path(paths["models"]) / "yolo26l.pt").write_bytes(b"base")

    return {
        "paths": paths,
        "detector": {
            "runtime_override_path": str(tmp_path / "data" / "ingest_model_selection.json"),
            "player_runtime_override_path": str(tmp_path / "data" / "ingest_player_model_selection.json"),
        },
        "watcher": {
            "extensions": [".mp4", ".insv", ".mov"],
            "ignore_suffixes": [".uploading", ".tmp", ".part"],
            "processed_state_file": "watcher_processed_ingest.json",
        },
        "dashboard": {
            "db_path": str(tmp_path / "dashboard.db"),
        },
    }


@pytest.fixture
def client(store, dashboard_config):
    with patch("src.dashboard.create_event_store", return_value=store):
        app = create_app(dashboard_config)
    return _ASGIClient(app)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class _ASGIClient:
    def __init__(self, app):
        self.app = app

    def request(self, method: str, url: str, **kwargs):
        async def _send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, url, **kwargs)

        return anyio.run(_send)

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)


class TestDashboardAPI:
    def test_index_returns_html(self, client):
        """GET / returns HTML content."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Soccer360" in resp.text

    def test_index_includes_detection_settings_nav_link(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert 'href="/settings/detection"' in resp.text

    def test_detection_settings_page_returns_readonly_html(self, client):
        resp = client.get("/settings/detection")
        assert resp.status_code == 200
        assert "Detection Settings" in resp.text
        assert "Readonly view" in resp.text
        assert "Ingest model selection is managed in Staging" in resp.text
        assert "Save Detection Settings" not in resp.text

    def test_status_idle(self, client):
        """GET /api/status returns idle when nothing is processing."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"

    def test_status_processing(self, client, store):
        """GET /api/status reflects running job."""
        store.job_created("job1", "match.mp4")
        store.job_started("job1", mode="normal")
        store.phase_started("job1", "detection")

        resp = client.get("/api/status")
        data = resp.json()
        assert data["state"] == "processing"
        assert data["job_id"] == "job1"

    def test_list_jobs_empty(self, client):
        """GET /api/jobs returns empty list initially."""
        resp = client.get("/api/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_jobs(self, client, store):
        """GET /api/jobs returns jobs."""
        store.job_created("job1", "match.mp4")
        store.job_created("job2", "match2.mp4")
        resp = client.get("/api/jobs")
        data = resp.json()
        assert len(data) == 2

    def test_clear_history_purges_dashboard_job_records(self, client, store):
        store.job_created("job1", "match.mp4")
        store.job_started("job1", mode="normal")
        store.phase_started("job1", "detection")
        store.phase_completed("job1", "detection", duration_sec=1.0)
        store.record_gpu_snapshot("job1", "detection", {"gpu_pct": 50})
        decision_id = store.request_decision("job1", "confirm", "Continue?", ["yes", "no"], "yes", 30)
        store.resolve_decision(decision_id, "yes", status="approved")
        store.job_completed("job1")

        resp = client.post("/api/history/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["jobs_deleted"] == 1
        assert data["phase_events_deleted"] == 1
        assert data["metrics_snapshots_deleted"] == 1
        assert data["decisions_deleted"] == 1
        assert store.get_jobs(limit=10) == []
        assert store.get_phases("job1") == []
        assert store.get_decision(decision_id) is None

    def test_clear_history_blocks_active_jobs(self, client, store):
        store.job_created("job1", "match.mp4")

        resp = client.post("/api/history/clear")
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Cannot clear history while a pipeline job is queued or running."

    def test_get_job_detail(self, client, store):
        """GET /api/jobs/{id} returns job with phases."""
        store.job_created("job1", "match.mp4")
        store.job_started("job1")
        store.phase_started("job1", "detection")
        store.phase_completed("job1", "detection", duration_sec=10.5)

        resp = client.get("/api/jobs/job1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job"]["job_id"] == "job1"
        assert len(data["phases"]) == 1
        assert data["phases"][0]["duration_sec"] == 10.5

    def test_get_job_not_found(self, client):
        """GET /api/jobs/{id} returns 404 for unknown job."""
        resp = client.get("/api/jobs/nonexistent")
        assert resp.status_code == 404

    def test_gpu_endpoint(self, client):
        """GET /api/gpu returns data (may show unavailable)."""
        resp = client.get("/api/gpu")
        assert resp.status_code == 200

    def test_pending_decisions_empty(self, client):
        """GET /api/decisions/pending returns empty list."""
        resp = client.get("/api/decisions/pending")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_resolve_decision(self, client, store):
        """POST /api/decisions/{id}/resolve resolves a pending decision."""
        did = store.request_decision("job1", "confirm", "Continue?", ["yes", "no"], "yes", 60)

        resp = client.post(
            f"/api/decisions/{did}/resolve",
            json={"response": "yes", "status": "approved"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Verify it's resolved
        decision = store.get_decision(did)
        assert decision["status"] == "approved"

    def test_resolve_decision_not_found(self, client):
        """POST /api/decisions/999/resolve returns 404."""
        resp = client.post("/api/decisions/999/resolve", json={"response": "yes", "status": "approved"})
        assert resp.status_code == 404

    def test_resolve_already_resolved(self, client, store):
        """POST on already-resolved decision returns 409."""
        did = store.request_decision("job1", "confirm", "OK?", timeout_sec=60)
        store.resolve_decision(did, "yes", status="approved")

        resp = client.post(
            f"/api/decisions/{did}/resolve",
            json={"response": "no", "status": "rejected"},
        )
        assert resp.status_code == 409

    def test_resolve_invalid_status(self, client, store):
        """POST with invalid status returns 400."""
        did = store.request_decision("job1", "confirm", "OK?", timeout_sec=60)
        resp = client.post(
            f"/api/decisions/{did}/resolve",
            json={"response": "yes", "status": "invalid"},
        )
        assert resp.status_code == 400

    def test_staging_files_endpoint_lists_video_candidates(self, client, dashboard_config):
        staging_dir = Path(dashboard_config["paths"]["stagging"])
        (staging_dir / "queued.mp4").write_bytes(b"video")
        (staging_dir / "notes.txt").write_text("ignore", encoding="utf-8")
        (staging_dir / "partial.part").write_bytes(b"partial")

        resp = client.get("/api/staging/files")
        assert resp.status_code == 200
        data = resp.json()
        assert [item["name"] for item in data] == ["queued.mp4"]

    def test_staging_import_moves_file_to_ingest(self, client, dashboard_config):
        staging_dir = Path(dashboard_config["paths"]["stagging"])
        ingest_dir = Path(dashboard_config["paths"]["ingest"])
        source_path = staging_dir / "queued.mp4"
        source_path.write_bytes(b"video")

        resp = client.post("/api/staging/import", json={"filename": "queued.mp4"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "queued.mp4"
        assert not source_path.exists()
        assert (ingest_dir / "queued.mp4").exists()

    def test_staging_import_conflicts_with_existing_ingest_file(self, client, dashboard_config):
        staging_dir = Path(dashboard_config["paths"]["stagging"])
        ingest_dir = Path(dashboard_config["paths"]["ingest"])
        (staging_dir / "queued.mp4").write_bytes(b"video")
        (ingest_dir / "queued.mp4").write_bytes(b"existing")

        resp = client.post("/api/staging/import", json={"filename": "queued.mp4"})
        assert resp.status_code == 409

    def test_staging_upload_saves_video_file(self, client, dashboard_config):
        staging_dir = Path(dashboard_config["paths"]["stagging"])

        resp = client.post(
            "/api/staging/upload",
            files={"file": ("uploaded.mp4", b"video-bytes", "video/mp4")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "uploaded.mp4"
        assert (staging_dir / "uploaded.mp4").read_bytes() == b"video-bytes"

    def test_staging_upload_rejects_unsupported_extension(self, client):
        resp = client.post(
            "/api/staging/upload",
            files={"file": ("notes.txt", b"not-a-video", "text/plain")},
        )

        assert resp.status_code == 400

    def test_staging_upload_rejects_existing_filename(self, client, dashboard_config):
        staging_dir = Path(dashboard_config["paths"]["stagging"])
        (staging_dir / "uploaded.mp4").write_bytes(b"existing")

        resp = client.post(
            "/api/staging/upload",
            files={"file": ("uploaded.mp4", b"video-bytes", "video/mp4")},
        )

        assert resp.status_code == 409

    def test_upload_labels_replaces_existing_match_snapshot(self, client, dashboard_config, tmp_path):
        labeling_dir = Path(dashboard_config["paths"]["labeling"])
        match_dir = labeling_dir / "match_a"
        frames_dir = match_dir / "frames"
        labels_dir = match_dir / "labels"
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "frame_000001.jpg").write_bytes(b"frame-1")
        (frames_dir / "frame_000002.jpg").write_bytes(b"frame-2")
        (labels_dir / "frame_000999.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

        zip_path = tmp_path / "labels.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("frame_000001.txt", "0 0.5 0.5 0.1 0.1\n")

        with zip_path.open("rb") as fh:
            resp = client.post(
                "/api/training/upload-labels/match_a",
                files={"file": ("labels.zip", fh.read(), "application/zip")},
            )

        assert resp.status_code == 200
        assert sorted(path.name for path in labels_dir.glob("*.txt")) == ["frame_000001.txt"]

    def test_labeling_status_counts_unique_matched_frames(self, client, dashboard_config):
        labeling_dir = Path(dashboard_config["paths"]["labeling"])
        match_dir = labeling_dir / "match_a"
        frames_dir = match_dir / "frames"
        labels_dir = match_dir / "labels"
        frames_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "frame_000001.jpg").write_bytes(b"frame-1")
        (labels_dir / "frame_000001.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (labels_dir / "frame_000001_jpg.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (labels_dir / "classes.txt").write_text("ball\n", encoding="utf-8")

        resp = client.get("/api/training/labeling-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_labeled"] == 1
        assert data["matches"][0]["labeled"] == 1

    def test_reset_match_endpoint_deletes_family_and_restores_source(
        self, client, store, dashboard_config
    ):
        processed_dir = Path(dashboard_config["paths"]["processed"])
        highlights_dir = Path(dashboard_config["paths"]["highlights"])
        labeling_dir = Path(dashboard_config["paths"]["labeling"])
        archive_dir = Path(dashboard_config["paths"]["archive_raw"])
        staging_dir = Path(dashboard_config["paths"]["stagging"])
        ingest_dir = Path(dashboard_config["paths"]["ingest"])

        archived_video = archive_dir / "match_a_20260319_job1.mp4"
        archived_video.write_bytes(b"archived-video")

        ingest_source = ingest_dir / "match_a.mp4"
        ingest_source.write_bytes(b"processed-source")
        state_path = processed_dir / ".state" / "watcher_processed_ingest.json"
        dedupe_store = ProcessedIngestStore(state_path)
        fingerprint = {"size": 16, "mtime_ns": 1}
        dedupe_store.mark_processed(ingest_source, fingerprint, job_path="job1/match_a.mp4")
        ingest_source.unlink()

        for job_id in ("job1", "job2"):
            store.job_created(job_id, f"/scratch/work/{job_id}/match_a.mp4")
            store.job_started(job_id, mode="normal")
            store.phase_started(job_id, "detection")
            store.phase_completed(job_id, "detection", duration_sec=1.0)
            store.record_gpu_snapshot(job_id, "detection", {"gpu_pct": 50})
            decision_id = store.request_decision(job_id, "confirm", "Continue?", ["yes", "no"], "yes", 30)
            store.resolve_decision(decision_id, "yes", status="approved")
            store.job_completed(job_id)

        _write_json(
            processed_dir / "match_a" / "metadata.json",
            {
                "game_name": "match_a",
                "job_id": "job1",
                "ingest_source_path": str(ingest_source),
                "ingest_archived_path": str(archived_video),
                "mode": "normal",
            },
        )
        (processed_dir / "match_a" / "broadcast.mp4").write_bytes(b"broadcast")
        _write_json(
            processed_dir / "match_a_run1" / "metadata.json",
            {
                "game_name": "match_a",
                "job_id": "job2",
                "ingest_source_path": str(ingest_source),
                "ingest_archived_path": str(archive_dir / "missing.mp4"),
                "mode": "normal",
            },
        )
        (processed_dir / "match_a_run1" / "broadcast.mp4").write_bytes(b"broadcast")
        (highlights_dir / "match_a").mkdir(parents=True, exist_ok=True)
        (highlights_dir / "match_a" / "highlight_001.mp4").write_bytes(b"highlight")
        (highlights_dir / "match_a_run1").mkdir(parents=True, exist_ok=True)
        (highlights_dir / "match_a_run1" / "highlight_001.mp4").write_bytes(b"highlight")
        (labeling_dir / "match_a" / "frames").mkdir(parents=True, exist_ok=True)
        (labeling_dir / "match_a" / "frames" / "frame_000001.jpg").write_bytes(b"frame")
        (labeling_dir / "match_a" / "labels").mkdir(parents=True, exist_ok=True)
        (labeling_dir / "match_a" / "labels" / "frame_000001.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n",
            encoding="utf-8",
        )
        (labeling_dir / "dataset").mkdir(parents=True, exist_ok=True)
        (labeling_dir / "dataset" / "dataset.yaml").write_text("path: dataset\n", encoding="utf-8")

        resp = client.post("/api/media/matches/match_a_run1/reset")
        assert resp.status_code == 200
        data = resp.json()

        assert data["canonical_match"] == "match_a"
        assert data["deleted_processed_dirs_count"] == 2
        assert data["deleted_highlights_dirs_count"] == 2
        assert data["labeling_deleted"] is True
        assert data["dataset_invalidated"] is True
        assert data["purged_job_ids"] == ["job1", "job2"]
        assert data["purged_job_count"] == 2
        assert data["restored_staging_path"] == str(staging_dir / "match_a.mp4")
        assert data["warnings"] == []

        assert not (processed_dir / "match_a").exists()
        assert not (processed_dir / "match_a_run1").exists()
        assert not (highlights_dir / "match_a").exists()
        assert not (highlights_dir / "match_a_run1").exists()
        assert not (labeling_dir / "match_a").exists()
        assert not (labeling_dir / "dataset").exists()
        assert not archived_video.exists()
        assert (staging_dir / "match_a.mp4").exists()
        assert store.get_job("job1") is None
        assert store.get_job("job2") is None
        reloaded = ProcessedIngestStore(state_path)
        assert str(ingest_source.resolve(strict=False)) not in reloaded._entries

    def test_reset_match_endpoint_restores_original_name_with_collision_suffix(
        self, client, store, dashboard_config
    ):
        processed_dir = Path(dashboard_config["paths"]["processed"])
        archive_dir = Path(dashboard_config["paths"]["archive_raw"])
        staging_dir = Path(dashboard_config["paths"]["stagging"])
        original_name = "match_c_original.mp4"

        archived_video = archive_dir / "match_c_original_20260319_jobc.mp4"
        archived_video.write_bytes(b"archived-video")
        (staging_dir / original_name).write_bytes(b"existing-staged-video")

        _write_json(
            processed_dir / "match_c" / "metadata.json",
            {
                "game_name": "match_c",
                "job_id": "job_c",
                "ingest_source_path": str(Path(dashboard_config["paths"]["ingest"]) / original_name),
                "ingest_archived_path": str(archived_video),
            },
        )
        (processed_dir / "match_c" / "broadcast.mp4").write_bytes(b"broadcast")
        store.job_created("job_c", "/scratch/work/job_c/match_c.mp4")
        store.job_started("job_c", mode="normal")
        store.job_completed("job_c")

        resp = client.post("/api/media/matches/match_c/reset")
        assert resp.status_code == 200
        data = resp.json()

        assert data["restored_staging_path"] == str(staging_dir / "match_c_original_01.mp4")
        assert (staging_dir / "match_c_original.mp4").read_bytes() == b"existing-staged-video"
        assert (staging_dir / "match_c_original_01.mp4").read_bytes() == b"archived-video"

    def test_reset_match_endpoint_warns_when_no_restore_source(self, client, store, dashboard_config):
        processed_dir = Path(dashboard_config["paths"]["processed"])
        _write_json(
            processed_dir / "match_b" / "metadata.json",
            {
                "game_name": "match_b",
                "job_id": "job_b",
                "ingest_source_path": str(Path(dashboard_config["paths"]["ingest"]) / "match_b.mp4"),
                "ingest_archived_path": str(Path(dashboard_config["paths"]["archive_raw"]) / "missing.mp4"),
            },
        )
        (processed_dir / "match_b" / "broadcast.mp4").write_bytes(b"broadcast")
        store.job_created("job_b", "/scratch/work/job_b/match_b.mp4")
        store.job_started("job_b", mode="normal")
        store.job_completed("job_b")

        resp = client.post("/api/media/matches/match_b/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["restored_staging_path"] is None
        assert "No restorable source video found in archive or ingest." in data["warnings"]

    def test_reset_match_endpoint_blocks_active_job(self, client, store, dashboard_config):
        processed_dir = Path(dashboard_config["paths"]["processed"])
        _write_json(
            processed_dir / "match_c" / "metadata.json",
            {
                "game_name": "match_c",
                "job_id": "job_c_done",
            },
        )
        (processed_dir / "match_c" / "broadcast.mp4").write_bytes(b"broadcast")

        store.job_created("job_c_active", "/scratch/work/job_c_active/match_c.mp4")

        resp = client.post("/api/media/matches/match_c/reset")
        assert resp.status_code == 409

    def test_reset_match_endpoint_blocks_training_in_progress(self, client, dashboard_config):
        processed_dir = Path(dashboard_config["paths"]["processed"])
        labeling_dir = Path(dashboard_config["paths"]["labeling"])
        dataset_dir = labeling_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.yaml").write_text("path: dataset\n", encoding="utf-8")

        _write_json(
            processed_dir / "match_d" / "metadata.json",
            {
                "game_name": "match_d",
                "job_id": "job_d_done",
            },
        )
        (processed_dir / "match_d" / "broadcast.mp4").write_bytes(b"broadcast")

        allow_finish = threading.Event()

        def fake_run(cmd, **kwargs):
            allow_finish.wait(timeout=1.0)
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

        with patch("src.dashboard.subprocess.run", side_effect=fake_run):
            train_resp = client.post("/api/training/train", json={"epochs": 1})
            assert train_resp.status_code == 200

            resp = client.post("/api/media/matches/match_d/reset")
            assert resp.status_code == 409
            assert (
                resp.json()["detail"]
                == "Cannot remove a processed match while dataset build or training is in progress."
            )

            allow_finish.set()

    def test_detection_settings_api_returns_grouped_effective_values(self, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        configured_ingest = models_dir / "yolo26l.pt"
        configured_ingest.write_bytes(b"ingest")
        dashboard_config["detection"] = {
            "path": str(configured_ingest),
            "classes": [32, 0],
            "conf": 0.15,
            "img_size": 960,
        }
        dashboard_config["detector"].update(
            {
                "batch_size": 12,
                "resolution": [1920, 960],
                "process_every_n_frames": 1,
            }
        )
        dashboard_config["field_of_interest"] = {"center_mode": "fixed", "yaw_window_deg": 160}
        dashboard_config["filters"] = {"max_jump_px": 250}
        dashboard_config["tracking"] = {"require_persistence": 3}
        dashboard_config["center_of_play"] = {"ball_blend_weight": 0.15}
        dashboard_config["camera"] = {"default_fov": 90.0}
        dashboard_config["reframer"] = {"output_resolution": [1920, 1080]}
        dashboard_config["highlights"] = {"max_clips": 20}
        dashboard_config["active_learning"] = {"export_max_frames": 600}
        dashboard_config["detection"]["player_path"] = str(configured_ingest)

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)

        resp = local_client.get("/api/settings/detection")
        assert resp.status_code == 200
        data = resp.json()
        assert data["readonly"] is True
        assert data["scope"] == "future_ingest_jobs"
        assert data["note"] == "Ingest model selection is managed in the Staging section of the main dashboard."

        group_titles = [group["title"] for group in data["groups"]]
        assert group_titles == [
            "Ingest Models",
            "Detection",
            "Field of Interest",
            "Ball Stabilization / Filters",
            "Player Detection & Clustering",
            "Camera / Auto-Follow",
            "Reframer / Output",
            "Highlights",
            "Active Learning",
        ]

        ingest_group = data["groups"][0]
        assert any(
            field["config_path"] == "runtime.ball.resolved_path" and field["value"] == str(configured_ingest)
            for field in ingest_group["fields"]
        )
        assert any(
            field["config_path"] == "runtime.ball.resolved_source" and field["value"] == "detection.path"
            for field in ingest_group["fields"]
        )
        assert any(
            field["config_path"] == "runtime.player.resolved_path" and field["value"] == str(configured_ingest)
            for field in ingest_group["fields"]
        )

        detection_group = next(group for group in data["groups"] if group["title"] == "Detection")
        assert any(
            field["config_path"] == "detection.classes" and field["value"] == [32, 0]
            for field in detection_group["fields"]
        )
        assert any(
            field["config_path"] == "detector.batch_size" and field["value"] == 12
            for field in detection_group["fields"]
        )

        camera_group = next(group for group in data["groups"] if group["title"] == "Camera / Auto-Follow")
        assert any(
            field["config_path"] == "camera.default_fov" and field["value"] == 90.0
            for field in camera_group["fields"]
        )


class TestTrainingAPI:
    def test_labeling_status_empty(self, client):
        """GET /api/training/labeling-status returns empty when no labeling dir."""
        resp = client.get("/api/training/labeling-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "matches" in data
        assert "total_frames" in data

    def test_training_status_idle(self, client):
        """GET /api/training/status returns idle initially."""
        resp = client.get("/api/training/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"

    def test_models_endpoint(self, client):
        """GET /api/training/models returns list."""
        resp = client.get("/api/training/models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_models_endpoint_marks_active_and_configured_base(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        active_model = models_dir / "ball_best.pt"
        configured_model = models_dir / "custom.pt"
        configured_ingest = models_dir / "yolo26l.pt"
        active_model.write_bytes(b"active")
        configured_model.write_bytes(b"configured")
        configured_ingest.write_bytes(b"ingest")
        dashboard_config["model"] = {"base_model": str(configured_model)}
        dashboard_config["detection"] = {
            "path": str(configured_ingest),
            "player_path": str(configured_ingest),
        }

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)

        resp = local_client.get("/api/training/models")
        assert resp.status_code == 200
        data = resp.json()
        assert any(item["path"] == str(active_model) and item["is_active"] for item in data)
        assert any(item["path"] == str(configured_model) and item["is_configured_base"] for item in data)
        assert any(item["path"] == str(configured_ingest) and item["is_configured_inference"] for item in data)
        assert any(item["path"] == str(active_model) and item["can_delete"] is False for item in data)
        assert any(item["path"] == str(configured_model) and item["can_delete"] is False for item in data)
        assert any(item["path"] == str(configured_ingest) and item["can_delete"] is False for item in data)

    def test_models_endpoint_hides_per_run_best_and_last_artifacts(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        promoted_model = models_dir / "named_model.pt"
        run_best = models_dir / "ball_model_20260319_0034" / "weights" / "best.pt"
        run_last = models_dir / "ball_model_20260319_0034" / "weights" / "last.pt"
        promoted_model.write_bytes(b"named")
        run_best.parent.mkdir(parents=True, exist_ok=True)
        run_best.write_bytes(b"best")
        run_last.write_bytes(b"last")

        resp = client.get("/api/training/models")
        assert resp.status_code == 200
        data = resp.json()
        paths = {item["path"] for item in data}

        assert str(promoted_model) in paths
        assert str(run_best) not in paths
        assert str(run_last) not in paths

    def test_inference_model_status_defaults_to_config_resolution(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        configured_ingest = models_dir / "yolo26l.pt"
        configured_ingest.write_bytes(b"ingest")
        dashboard_config["detection"] = {
            "path": str(configured_ingest),
            "player_path": str(configured_ingest),
        }

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)

        resp = local_client.get("/api/inference/model")
        assert resp.status_code == 200
        data = resp.json()
        assert data["selection_mode"] == "config"
        assert data["resolved_path"] == str(configured_ingest)
        assert data["resolved_source"] == "detection.path"
        assert data["config_locked"] is False
        assert data["ball"]["resolved_path"] == str(configured_ingest)
        assert data["player"]["resolved_path"] == str(configured_ingest)
        assert data["dual_model_enabled"] is False

    def test_set_inference_model_to_pinned_updates_runtime_selection(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        configured_ingest = models_dir / "yolo26l.pt"
        selected_model = models_dir / "experiment.pt"
        configured_ingest.write_bytes(b"ingest")
        selected_model.write_bytes(b"selected")
        dashboard_config["detection"] = {"path": str(configured_ingest)}

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)

        resp = local_client.post(
            "/api/inference/model",
            json={"mode": "pinned", "path": str(selected_model)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["selection_mode"] == "pinned"
        assert data["selected_path"] == str(selected_model)
        assert data["resolved_path"] == str(selected_model)
        assert data["resolved_source"] == "runtime.pinned.ball"

        models_resp = local_client.get("/api/training/models")
        models_data = models_resp.json()
        assert any(item["path"] == str(selected_model) and item["is_ball_inference_active"] for item in models_data)
        assert any(item["path"] == str(selected_model) and item["can_delete"] is False for item in models_data)

    def test_set_inference_model_to_auto_prefers_ball_best(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        active_model = models_dir / "ball_best.pt"
        configured_ingest = models_dir / "yolo26l.pt"
        active_model.write_bytes(b"active")
        configured_ingest.write_bytes(b"ingest")
        dashboard_config["detection"] = {"path": str(configured_ingest)}

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)

        resp = local_client.post("/api/inference/model", json={"mode": "auto"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["selection_mode"] == "auto"
        assert data["resolved_path"] == str(active_model)
        assert data["resolved_source"] == "runtime.auto.ball"

    def test_set_inference_models_supports_dual_role_selection(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        ball_model = models_dir / "ball_only.pt"
        player_model = models_dir / "player_base.pt"
        ball_model.write_bytes(b"ball")
        player_model.write_bytes(b"player")

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)

        resp = local_client.post(
            "/api/inference/model",
            json={
                "ball": {"mode": "pinned", "path": str(ball_model)},
                "player": {"mode": "pinned", "path": str(player_model)},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ball"]["resolved_path"] == str(ball_model)
        assert data["ball"]["resolved_source"] == "runtime.pinned.ball"
        assert data["player"]["resolved_path"] == str(player_model)
        assert data["player"]["resolved_source"] == "runtime.pinned.player"
        assert data["dual_model_enabled"] is True

    def test_set_inference_model_rejects_when_locked_by_detector_config(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        locked_model = models_dir / "locked.pt"
        selected_model = models_dir / "selected.pt"
        locked_model.write_bytes(b"locked")
        selected_model.write_bytes(b"selected")
        dashboard_config["detector"] = {
            "model_path": str(locked_model),
            "runtime_override_path": str(Path(dashboard_config["paths"]["processed"]).parent / "data" / "ingest_model_selection.json"),
        }

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)

        resp = local_client.post(
            "/api/inference/model",
            json={"mode": "pinned", "path": str(selected_model)},
        )
        assert resp.status_code == 409

    def test_training_endpoint_passes_selected_base_model(self, client, dashboard_config):
        labeling_dir = Path(dashboard_config["paths"]["labeling"])
        models_dir = Path(dashboard_config["paths"]["models"])
        dataset_dir = labeling_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.yaml").write_text("path: dataset\n", encoding="utf-8")
        selected_model = models_dir / "custom.pt"
        selected_model.write_bytes(b"weights")

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

        with patch("src.dashboard.subprocess.run", side_effect=fake_run):
            resp = client.post(
                "/api/training/train",
                json={"epochs": 7, "base_model": str(selected_model)},
            )

        assert resp.status_code == 200
        assert "--base-model" in captured["cmd"]
        assert str(selected_model) in captured["cmd"]

    def test_training_endpoint_passes_named_output_model(self, client, dashboard_config):
        labeling_dir = Path(dashboard_config["paths"]["labeling"])
        models_dir = Path(dashboard_config["paths"]["models"])
        dataset_dir = labeling_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.yaml").write_text("path: dataset\n", encoding="utf-8")
        selected_model = models_dir / "custom.pt"
        selected_model.write_bytes(b"weights")

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

        with patch("src.dashboard.subprocess.run", side_effect=fake_run):
            resp = client.post(
                "/api/training/train",
                json={
                    "epochs": 7,
                    "base_model": str(selected_model),
                    "output_model_name": "experiment_a",
                    "update_active": False,
                },
            )

        assert resp.status_code == 200
        assert "--output-model-name" in captured["cmd"]
        assert "experiment_a" in captured["cmd"]
        assert "--no-update-active" in captured["cmd"]

    def test_delete_old_model_endpoint_removes_local_model_file(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        stale_model = models_dir / "stale_model.pt"
        stale_model.write_bytes(b"old")

        resp = client.post("/api/training/models/delete", json={"path": str(stale_model)})

        assert resp.status_code == 200
        assert not stale_model.exists()

    def test_delete_model_endpoint_rejects_active_model(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        active_model = models_dir / "ball_best.pt"
        active_model.write_bytes(b"active")

        resp = client.post("/api/training/models/delete", json={"path": str(active_model)})

        assert resp.status_code == 409
        assert active_model.exists()

    def test_delete_model_endpoint_rejects_configured_ingest_model(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        configured_ingest = models_dir / "yolo26l.pt"
        configured_ingest.write_bytes(b"ingest")
        dashboard_config["detection"] = {"path": str(configured_ingest)}

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)

        resp = local_client.post("/api/training/models/delete", json={"path": str(configured_ingest)})

        assert resp.status_code == 409
        assert configured_ingest.exists()

    def test_delete_model_endpoint_rejects_runtime_selected_ingest_model(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        configured_ingest = models_dir / "yolo26l.pt"
        selected_model = models_dir / "experiment.pt"
        configured_ingest.write_bytes(b"ingest")
        selected_model.write_bytes(b"selected")
        dashboard_config["detection"] = {"path": str(configured_ingest)}

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)
        set_resp = local_client.post(
            "/api/inference/model",
            json={"mode": "pinned", "path": str(selected_model)},
        )
        assert set_resp.status_code == 200

        resp = local_client.post("/api/training/models/delete", json={"path": str(selected_model)})

        assert resp.status_code == 409
        assert selected_model.exists()

    def test_delete_model_endpoint_rejects_runtime_selected_player_model(self, client, dashboard_config):
        models_dir = Path(dashboard_config["paths"]["models"])
        selected_model = models_dir / "player_experiment.pt"
        selected_model.write_bytes(b"selected")

        with patch("src.dashboard.create_event_store", return_value=EventStore(":memory:")):
            app = create_app(dashboard_config)
        local_client = _ASGIClient(app)
        set_resp = local_client.post(
            "/api/inference/model",
            json={"player": {"mode": "pinned", "path": str(selected_model)}},
        )
        assert set_resp.status_code == 200

        resp = local_client.post("/api/training/models/delete", json={"path": str(selected_model)})

        assert resp.status_code == 409
        assert selected_model.exists()

    def test_build_dataset_helper_creates_dataset(self, tmp_path: Path):
        """Dataset helper should build train/val splits from labeled frames."""
        labeling_dir = tmp_path / "labeling"
        for match_name in ("match_a", "match_b"):
            frames_dir = labeling_dir / match_name / "frames"
            labels_dir = labeling_dir / match_name / "labels"
            frames_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            for frame_idx in (1, 2):
                stem = f"frame_{frame_idx:06d}"
                (frames_dir / f"{stem}.jpg").write_bytes(b"fake-jpeg-data")
                (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        result = _build_dataset_from_labels(labeling_dir=labeling_dir, val_ratio=0.5)

        assert result["train_count"] == 2
        assert result["val_count"] == 2
        assert result["total_count"] == 4
        assert result["match_counts"] == {"match_a": 2, "match_b": 2}
        assert result["dataset_yaml"].exists()
        assert (labeling_dir / "dataset" / "train" / "images").is_dir()
        assert (labeling_dir / "dataset" / "val" / "images").is_dir()

    def test_build_dataset_helper_normalizes_common_labelstudio_names(self, tmp_path: Path):
        labeling_dir = tmp_path / "labeling"
        frames_dir = labeling_dir / "match_a" / "frames"
        labels_dir = labeling_dir / "match_a" / "labels"
        frames_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        (frames_dir / "frame_000001.jpg").write_bytes(b"fake-jpeg-data")
        (labels_dir / "frame_000001_jpg.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (labels_dir / "classes.txt").write_text("ball\n", encoding="utf-8")

        result = _build_dataset_from_labels(labeling_dir=labeling_dir, val_ratio=0.5)

        assert result["total_count"] == 1
        built_labels = list((labeling_dir / "dataset" / "train" / "labels").glob("*.txt"))
        if not built_labels:
            built_labels = list((labeling_dir / "dataset" / "val" / "labels").glob("*.txt"))
        assert len(built_labels) == 1

    def test_build_dataset_helper_dedupes_multiple_label_files_for_same_frame(self, tmp_path: Path):
        labeling_dir = tmp_path / "labeling"
        frames_dir = labeling_dir / "match_a" / "frames"
        labels_dir = labeling_dir / "match_a" / "labels"
        frames_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        (frames_dir / "frame_000001.jpg").write_bytes(b"fake-jpeg-data")
        (labels_dir / "frame_000001.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (labels_dir / "frame_000001_jpg.txt").write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")

        result = _build_dataset_from_labels(labeling_dir=labeling_dir, val_ratio=0.5)

        assert result["total_count"] == 1
