"""Tests for the FastAPI dashboard application."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.dashboard import create_app
from src.events import EventStore


@pytest.fixture
def store():
    return EventStore(":memory:")


@pytest.fixture
def client(store):
    with patch("src.dashboard.create_event_store", return_value=store):
        app = create_app({})
    return TestClient(app)


class TestDashboardAPI:
    def test_index_returns_html(self, client):
        """GET / returns HTML content."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Soccer360" in resp.text

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
