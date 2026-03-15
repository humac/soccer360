"""Event bus and SQLite state store for the monitoring dashboard.

The pipeline worker writes events to SQLite; the dashboard reads them
and streams to the browser via SSE.  SQLite WAL mode allows concurrent
readers with a single writer safely.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("soccer360.events")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT UNIQUE NOT NULL,
    input_path  TEXT,
    status      TEXT NOT NULL DEFAULT 'queued',   -- queued | running | completed | failed
    mode        TEXT,
    started_at  TEXT,
    completed_at TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS phase_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    phase_name  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',  -- running | completed | failed
    started_at  TEXT,
    completed_at TEXT,
    duration_sec REAL,
    stats_json  TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT,
    phase_name  TEXT,
    timestamp   TEXT NOT NULL,
    gpu_json    TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT,
    decision_type   TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    options_json    TEXT,
    default_option  TEXT,
    timeout_sec     INTEGER NOT NULL DEFAULT 60,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected | timeout
    response        TEXT,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_phase_events_job ON phase_events(job_id);
CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);
"""


class EventStore:
    """Thread-safe SQLite store for pipeline events.

    For file-based databases, uses per-thread connections with WAL mode.
    For in-memory databases, uses a shared connection with serialized access.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._is_memory = self._db_path == ":memory:"
        self._lock = threading.Lock()

        if self._is_memory:
            # Single shared connection for in-memory (test) databases
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.executescript(_SCHEMA)
            self._shared_conn.commit()
        else:
            self._local = threading.local()
            self._shared_conn = None
            # Initialize schema on first connection
            conn = self._conn()
            conn.executescript(_SCHEMA)
            conn.commit()

    def _conn(self) -> sqlite3.Connection:
        """Get a connection (shared for memory, per-thread for files)."""
        if self._is_memory:
            return self._shared_conn

        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def close(self):
        if self._is_memory:
            if self._shared_conn:
                self._shared_conn.close()
                self._shared_conn = None
            return
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _execute_write(self, sql: str, params: tuple = ()) -> int:
        """Execute a write statement with proper locking. Returns lastrowid."""
        with self._lock:
            conn = self._conn()
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.lastrowid

    @contextmanager
    def _read_lock(self):
        """Acquire lock for reads on shared in-memory connections."""
        if self._is_memory:
            with self._lock:
                yield self._shared_conn
        else:
            yield self._conn()

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def job_created(self, job_id: str, input_path: str) -> int:
        now = _now_iso()
        return self._execute_write(
            "INSERT INTO jobs (job_id, input_path, status, created_at) VALUES (?, ?, 'queued', ?)",
            (job_id, input_path, now),
        )

    def job_started(self, job_id: str, mode: str | None = None):
        now = _now_iso()
        self._execute_write(
            "UPDATE jobs SET status='running', mode=?, started_at=? WHERE job_id=?",
            (mode, now, job_id),
        )

    def job_completed(self, job_id: str):
        now = _now_iso()
        self._execute_write(
            "UPDATE jobs SET status='completed', completed_at=? WHERE job_id=?",
            (now, job_id),
        )

    def job_failed(self, job_id: str, error: str | None = None):
        now = _now_iso()
        self._execute_write(
            "UPDATE jobs SET status='failed', completed_at=?, error=? WHERE job_id=?",
            (now, error, job_id),
        )

    # ------------------------------------------------------------------
    # Phase lifecycle
    # ------------------------------------------------------------------

    def phase_started(self, job_id: str, phase_name: str) -> int:
        now = _now_iso()
        return self._execute_write(
            "INSERT INTO phase_events (job_id, phase_name, status, started_at, created_at) "
            "VALUES (?, ?, 'running', ?, ?)",
            (job_id, phase_name, now, now),
        )

    def phase_completed(
        self,
        job_id: str,
        phase_name: str,
        duration_sec: float | None = None,
        stats: dict | None = None,
    ):
        now = _now_iso()
        stats_json = json.dumps(stats) if stats else None
        self._execute_write(
            "UPDATE phase_events SET status='completed', completed_at=?, duration_sec=?, stats_json=? "
            "WHERE job_id=? AND phase_name=? AND status='running'",
            (now, duration_sec, stats_json, job_id, phase_name),
        )

    def phase_failed(self, job_id: str, phase_name: str, error: str | None = None):
        now = _now_iso()
        self._execute_write(
            "UPDATE phase_events SET status='failed', completed_at=?, stats_json=? "
            "WHERE job_id=? AND phase_name=? AND status='running'",
            (now, json.dumps({"error": error}) if error else None, job_id, phase_name),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def record_gpu_snapshot(self, job_id: str | None, phase_name: str | None, gpu_data: dict | None):
        now = _now_iso()
        self._execute_write(
            "INSERT INTO metrics_snapshots (job_id, phase_name, timestamp, gpu_json) VALUES (?, ?, ?, ?)",
            (job_id, phase_name, now, json.dumps(gpu_data) if gpu_data else None),
        )

    # ------------------------------------------------------------------
    # Decision queue
    # ------------------------------------------------------------------

    def request_decision(
        self,
        job_id: str | None,
        decision_type: str,
        prompt: str,
        options: list[str] | None = None,
        default_option: str | None = None,
        timeout_sec: int = 60,
    ) -> int:
        now = _now_iso()
        return self._execute_write(
            "INSERT INTO decisions (job_id, decision_type, prompt, options_json, default_option, "
            "timeout_sec, status, created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (
                job_id,
                decision_type,
                prompt,
                json.dumps(options) if options else None,
                default_option,
                timeout_sec,
                now,
            ),
        )

    def resolve_decision(self, decision_id: int, response: str, status: str = "approved"):
        now = _now_iso()
        self._execute_write(
            "UPDATE decisions SET status=?, response=?, resolved_at=? WHERE id=? AND status='pending'",
            (status, response, now, decision_id),
        )

    def get_decision(self, decision_id: int) -> dict | None:
        with self._read_lock() as conn:
            row = conn.execute("SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()
            return dict(row) if row else None

    def get_pending_decisions(self) -> list[dict]:
        with self._read_lock() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE status='pending' ORDER BY created_at"
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Queries (read-only, used by dashboard)
    # ------------------------------------------------------------------

    def get_jobs(self, limit: int = 50) -> list[dict]:
        with self._read_lock() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_job(self, job_id: str) -> dict | None:
        with self._read_lock() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_phases(self, job_id: str) -> list[dict]:
        with self._read_lock() as conn:
            rows = conn.execute(
                "SELECT * FROM phase_events WHERE job_id=? ORDER BY id", (job_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_current_status(self) -> dict:
        """Return a summary of the current pipeline state."""
        with self._read_lock() as conn:
            running = conn.execute(
                "SELECT * FROM jobs WHERE status='running' ORDER BY id DESC LIMIT 1"
            ).fetchone()

            queued_count = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status='queued'"
            ).fetchone()[0]

            if running:
                current_phase = conn.execute(
                    "SELECT * FROM phase_events WHERE job_id=? AND status='running' ORDER BY id DESC LIMIT 1",
                    (running["job_id"],),
                ).fetchone()
                completed_phases = conn.execute(
                    "SELECT COUNT(*) FROM phase_events WHERE job_id=? AND status='completed'",
                    (running["job_id"],),
                ).fetchone()[0]
                return {
                    "state": "processing",
                    "job_id": running["job_id"],
                    "input_path": running["input_path"],
                    "mode": running["mode"],
                    "current_phase": dict(current_phase) if current_phase else None,
                    "completed_phases": completed_phases,
                    "queued": queued_count,
                }

            return {"state": "idle", "queued": queued_count}

    def get_events_since(self, after_id: int = 0, limit: int = 100) -> list[dict]:
        """Get recent phase events with id > after_id for SSE streaming."""
        with self._read_lock() as conn:
            rows = conn.execute(
                "SELECT * FROM phase_events WHERE id > ? ORDER BY id LIMIT ?",
                (after_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_latest_event_id(self) -> int:
        with self._read_lock() as conn:
            row = conn.execute("SELECT MAX(id) FROM phase_events").fetchone()
            return row[0] or 0


class EventBus:
    """High-level interface used by the pipeline to emit events.

    Wraps EventStore with convenience methods and null-safety.
    """

    def __init__(self, store: EventStore):
        self.store = store

    def job_created(self, job_id: str, input_path: str):
        try:
            self.store.job_created(job_id, str(input_path))
        except Exception:
            logger.warning("EventBus: failed to record job_created", exc_info=True)

    def job_started(self, job_id: str, mode: str | None = None):
        try:
            self.store.job_started(job_id, mode)
        except Exception:
            logger.warning("EventBus: failed to record job_started", exc_info=True)

    def job_completed(self, job_id: str):
        try:
            self.store.job_completed(job_id)
        except Exception:
            logger.warning("EventBus: failed to record job_completed", exc_info=True)

    def job_failed(self, job_id: str, error: str | None = None):
        try:
            self.store.job_failed(job_id, error)
        except Exception:
            logger.warning("EventBus: failed to record job_failed", exc_info=True)

    def phase_started(self, job_id: str, phase_name: str):
        try:
            self.store.phase_started(job_id, phase_name)
        except Exception:
            logger.warning("EventBus: failed to record phase_started", exc_info=True)

    def phase_completed(
        self, job_id: str, phase_name: str, duration_sec: float | None = None, stats: dict | None = None
    ):
        try:
            self.store.phase_completed(job_id, phase_name, duration_sec, stats)
        except Exception:
            logger.warning("EventBus: failed to record phase_completed", exc_info=True)

    def phase_failed(self, job_id: str, phase_name: str, error: str | None = None):
        try:
            self.store.phase_failed(job_id, phase_name, error)
        except Exception:
            logger.warning("EventBus: failed to record phase_failed", exc_info=True)

    def record_gpu_snapshot(self, job_id: str | None, phase_name: str | None, gpu_data: dict | None):
        try:
            self.store.record_gpu_snapshot(job_id, phase_name, gpu_data)
        except Exception:
            logger.warning("EventBus: failed to record gpu_snapshot", exc_info=True)

    def request_decision(
        self,
        job_id: str | None,
        decision_type: str,
        prompt: str,
        options: list[str] | None = None,
        default_option: str | None = None,
        timeout_sec: int = 60,
    ) -> str:
        """Request a decision and block until resolved or timeout.

        Returns the response string (or default_option on timeout).
        """
        try:
            decision_id = self.store.request_decision(
                job_id, decision_type, prompt, options, default_option, timeout_sec
            )
        except Exception:
            logger.warning("EventBus: failed to create decision request", exc_info=True)
            return default_option or ""

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                decision = self.store.get_decision(decision_id)
                if decision and decision["status"] != "pending":
                    return decision.get("response") or default_option or ""
            except Exception:
                logger.warning("EventBus: failed to poll decision", exc_info=True)
            time.sleep(0.5)

        # Timeout — resolve with default
        try:
            self.store.resolve_decision(decision_id, default_option or "", status="timeout")
        except Exception:
            pass
        logger.info("Decision %d timed out after %ds, using default: %s", decision_id, timeout_sec, default_option)
        return default_option or ""


def create_event_store(config: dict) -> EventStore:
    """Create an EventStore from pipeline config."""
    dashboard_cfg = config.get("dashboard", {})
    db_path = dashboard_cfg.get("db_path", "/tank/data/dashboard.db")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return EventStore(db_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
