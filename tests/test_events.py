"""Tests for EventStore and EventBus."""

from __future__ import annotations

import time
import threading
from pathlib import Path

from src.events import EventBus, EventStore


class TestEventStore:
    def test_job_lifecycle(self):
        """EventStore tracks job creation, start, and completion."""
        store = EventStore(":memory:")
        store.job_created("job1", "/tank/ingest/match.mp4")
        store.job_started("job1", mode="normal")
        store.job_completed("job1")

        job = store.get_job("job1")
        assert job["status"] == "completed"
        assert job["mode"] == "normal"
        assert job["input_path"] == "/tank/ingest/match.mp4"

    def test_job_failure(self):
        """EventStore records job failure with error message."""
        store = EventStore(":memory:")
        store.job_created("job1", "/tank/ingest/match.mp4")
        store.job_started("job1")
        store.job_failed("job1", error="GPU OOM")

        job = store.get_job("job1")
        assert job["status"] == "failed"
        assert job["error"] == "GPU OOM"

    def test_phase_lifecycle(self):
        """EventStore tracks phase start and completion."""
        store = EventStore(":memory:")
        store.job_created("job1", "match.mp4")
        store.phase_started("job1", "detection")
        store.phase_completed("job1", "detection", duration_sec=45.2, stats={"count": 100})

        phases = store.get_phases("job1")
        assert len(phases) == 1
        assert phases[0]["phase_name"] == "detection"
        assert phases[0]["status"] == "completed"
        assert phases[0]["duration_sec"] == 45.2

    def test_phase_failure(self):
        """EventStore tracks phase failure."""
        store = EventStore(":memory:")
        store.job_created("job1", "match.mp4")
        store.phase_started("job1", "detection")
        store.phase_failed("job1", "detection", error="model not found")

        phases = store.get_phases("job1")
        assert len(phases) == 1
        assert phases[0]["status"] == "failed"

    def test_multiple_phases(self):
        """EventStore tracks multiple phases for a job."""
        store = EventStore(":memory:")
        store.job_created("job1", "match.mp4")
        for phase in ["detection", "tracking", "camera"]:
            store.phase_started("job1", phase)
            store.phase_completed("job1", phase, duration_sec=1.0)

        phases = store.get_phases("job1")
        assert len(phases) == 3

    def test_get_jobs_ordered(self):
        """Jobs are returned in reverse creation order."""
        store = EventStore(":memory:")
        store.job_created("job1", "match1.mp4")
        store.job_created("job2", "match2.mp4")
        store.job_created("job3", "match3.mp4")

        jobs = store.get_jobs(limit=10)
        assert [j["job_id"] for j in jobs] == ["job3", "job2", "job1"]

    def test_get_jobs_limit(self):
        """Jobs list respects limit parameter."""
        store = EventStore(":memory:")
        for i in range(10):
            store.job_created(f"job{i}", f"match{i}.mp4")

        jobs = store.get_jobs(limit=3)
        assert len(jobs) == 3

    def test_get_current_status_idle(self):
        """Status reports idle when no running jobs."""
        store = EventStore(":memory:")
        status = store.get_current_status()
        assert status["state"] == "idle"
        assert status["queued"] == 0

    def test_get_current_status_processing(self):
        """Status reports processing with current phase info."""
        store = EventStore(":memory:")
        store.job_created("job1", "match.mp4")
        store.job_started("job1", mode="normal")
        store.phase_started("job1", "detection")

        status = store.get_current_status()
        assert status["state"] == "processing"
        assert status["job_id"] == "job1"
        assert status["current_phase"]["phase_name"] == "detection"

    def test_get_events_since(self):
        """Events since a given ID returns only newer events."""
        store = EventStore(":memory:")
        store.job_created("job1", "match.mp4")
        store.phase_started("job1", "detection")
        store.phase_completed("job1", "detection", duration_sec=10.0)
        store.phase_started("job1", "tracking")

        # phase_started creates rows; phase_completed updates existing row
        all_events = store.get_events_since(after_id=0)
        assert len(all_events) == 2  # detection (started then completed=update), tracking (started)

        first_id = all_events[0]["id"]
        later = store.get_events_since(after_id=first_id)
        assert len(later) == 1  # only tracking

    def test_gpu_snapshot(self):
        """GPU snapshots are recorded."""
        store = EventStore(":memory:")
        store.record_gpu_snapshot("job1", "detection", {"gpu_pct": 85})
        # No assertion on retrieval since we don't have a dedicated query yet.
        # Just verify it doesn't raise.

    def test_get_active_jobs_by_input_stem(self):
        """Queued/running jobs can be filtered by input stem."""
        store = EventStore(":memory:")
        store.job_created("job1", "/scratch/work/job1/match_a.mp4")
        store.job_created("job2", "/scratch/work/job2/match_b.mp4")
        store.job_started("job2", mode="normal")
        store.job_created("job3", "/scratch/work/job3/other.mp4")
        store.job_started("job3", mode="normal")
        store.job_completed("job3")

        matches = store.get_active_jobs_by_input_stem("match_b")
        assert [job["job_id"] for job in matches] == ["job2"]

    def test_delete_jobs_purges_related_rows(self):
        """Deleting jobs should remove all related event/history rows."""
        store = EventStore(":memory:")
        store.job_created("job1", "/scratch/work/job1/match.mp4")
        store.job_started("job1", mode="normal")
        store.phase_started("job1", "detection")
        store.phase_completed("job1", "detection", duration_sec=1.0)
        store.record_gpu_snapshot("job1", "detection", {"gpu_pct": 50})
        decision_id = store.request_decision("job1", "confirm", "Continue?", ["yes", "no"], "yes", 30)
        store.resolve_decision(decision_id, "yes", status="approved")
        store.job_completed("job1")

        deleted = store.delete_jobs(["job1"])
        assert deleted == 1
        assert store.get_job("job1") is None
        assert store.get_phases("job1") == []
        assert store.get_decision(decision_id) is None
        assert store.get_jobs(limit=10) == []

    def test_clear_history_purges_all_related_rows(self):
        """Clearing history removes jobs and all dependent history tables."""
        store = EventStore(":memory:")
        store.job_created("job1", "/scratch/work/job1/match.mp4")
        store.job_started("job1", mode="normal")
        store.phase_started("job1", "detection")
        store.phase_completed("job1", "detection", duration_sec=1.0)
        store.record_gpu_snapshot("job1", "detection", {"gpu_pct": 50})
        decision_id = store.request_decision("job1", "confirm", "Continue?", ["yes", "no"], "yes", 30)
        store.resolve_decision(decision_id, "yes", status="approved")
        store.job_completed("job1")

        summary = store.clear_history()

        assert summary == {
            "jobs_deleted": 1,
            "phase_events_deleted": 1,
            "metrics_snapshots_deleted": 1,
            "decisions_deleted": 1,
        }
        assert store.get_jobs(limit=10) == []
        assert store.get_job("job1") is None
        assert store.get_phases("job1") == []
        assert store.get_decision(decision_id) is None

    def test_decision_lifecycle(self):
        """Decisions can be created and resolved."""
        store = EventStore(":memory:")
        did = store.request_decision("job1", "mode_confirm", "Continue?", ["yes", "no"], "yes", 30)
        assert did > 0

        pending = store.get_pending_decisions()
        assert len(pending) == 1
        assert pending[0]["prompt"] == "Continue?"

        store.resolve_decision(did, "yes", status="approved")
        decision = store.get_decision(did)
        assert decision["status"] == "approved"
        assert decision["response"] == "yes"

        # No more pending
        assert len(store.get_pending_decisions()) == 0

    def test_decision_already_resolved(self):
        """Resolving an already-resolved decision is a no-op."""
        store = EventStore(":memory:")
        did = store.request_decision("job1", "confirm", "OK?", timeout_sec=10)
        store.resolve_decision(did, "yes", status="approved")
        # Second resolve should not change anything
        store.resolve_decision(did, "no", status="rejected")
        decision = store.get_decision(did)
        assert decision["status"] == "approved"

    def test_thread_safety(self):
        """EventStore handles concurrent writes from multiple threads."""
        store = EventStore(":memory:")
        errors = []

        def writer(thread_id):
            try:
                for i in range(10):
                    job_id = f"t{thread_id}_j{i}"
                    store.job_created(job_id, f"match_{job_id}.mp4")
                    store.job_started(job_id)
                    store.phase_started(job_id, "detection")
                    store.phase_completed(job_id, "detection", duration_sec=0.1)
                    store.job_completed(job_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        jobs = store.get_jobs(limit=100)
        assert len(jobs) == 40  # 4 threads * 10 jobs

    def test_file_store_cleans_up_stale_jobs_by_default(self, tmp_path: Path):
        db_path = tmp_path / "events.db"
        store = EventStore(db_path, cleanup_stale_jobs=False)
        store.job_created("job1", "match.mp4")
        store.job_started("job1", mode="normal")
        store.phase_started("job1", "detection")
        store.close()

        reopened = EventStore(db_path)
        job = reopened.get_job("job1")
        phases = reopened.get_phases("job1")

        assert job["status"] == "failed"
        assert job["error"] == "Abandoned: service restarted"
        assert phases[0]["status"] == "failed"

    def test_file_store_can_skip_stale_job_cleanup(self, tmp_path: Path):
        db_path = tmp_path / "events.db"
        store = EventStore(db_path, cleanup_stale_jobs=False)
        store.job_created("job1", "match.mp4")
        store.job_started("job1", mode="normal")
        store.phase_started("job1", "detection")
        store.close()

        reopened = EventStore(db_path, cleanup_stale_jobs=False)
        job = reopened.get_job("job1")
        phases = reopened.get_phases("job1")

        assert job["status"] == "running"
        assert job["error"] is None
        assert phases[0]["status"] == "running"


class TestEventBus:
    def test_null_safety(self):
        """EventBus never raises even if store has issues."""
        store = EventStore(":memory:")
        bus = EventBus(store)

        # These should all succeed without raising
        bus.job_created("j1", "match.mp4")
        bus.job_started("j1", mode="normal")
        bus.phase_started("j1", "detection")
        bus.phase_completed("j1", "detection", duration_sec=1.0)
        bus.phase_failed("j1", "tracking", error="oops")
        bus.job_completed("j1")
        bus.job_failed("j2", error="boom")
        bus.record_gpu_snapshot("j1", "detection", {"gpu_pct": 50})

    def test_decision_timeout(self):
        """EventBus.request_decision returns default on timeout."""
        store = EventStore(":memory:")
        bus = EventBus(store)

        start = time.monotonic()
        result = bus.request_decision("j1", "confirm", "Continue?", default_option="yes", timeout_sec=1)
        elapsed = time.monotonic() - start

        assert result == "yes"
        assert elapsed >= 0.9  # Should have waited ~1 second

    def test_decision_resolved_before_timeout(self):
        """EventBus.request_decision returns early when resolved."""
        store = EventStore(":memory:")
        bus = EventBus(store)

        # Pre-create the decision and resolve it quickly in another thread
        def resolve_soon():
            time.sleep(0.3)
            pending = store.get_pending_decisions()
            if pending:
                store.resolve_decision(pending[0]["id"], "go_ahead", status="approved")

        t = threading.Thread(target=resolve_soon)
        t.start()

        start = time.monotonic()
        result = bus.request_decision("j1", "confirm", "OK?", default_option="yes", timeout_sec=10)
        elapsed = time.monotonic() - start

        t.join()
        assert result == "go_ahead"
        assert elapsed < 5  # Should return well before 10s timeout
