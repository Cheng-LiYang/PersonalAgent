"""Persistent, bounded execution of cooperative background jobs.

``ThreadPoolExecutor`` deliberately uses an unbounded internal queue.  This
module places a semaphore in front of it so the number of running plus waiting
jobs is bounded, while SQLite remains the source of truth for externally
visible state and progress.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
JOB_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES


class QueueFullError(RuntimeError):
    """Raised when all worker and queue slots are occupied."""


class ManagerClosedError(RuntimeError):
    """Raised when submitting work after manager shutdown."""


class JobCancelled(RuntimeError):
    """Workers may raise this after observing their cancellation predicate."""


JobWorker = Callable[[Callable[..., bool], Callable[[], bool]], Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _as_utc(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    else:
        parsed = value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class PersistentJobManager:
    """Run bounded background work with durable status and progress history."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_workers: int = 4,
        queue_capacity: int = 32,
        completion_ttl_seconds: float | None = 24 * 60 * 60,
        preview_chars: int = 4_096,
        max_payload_chars: int = 1_000_000,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if queue_capacity < 0:
            raise ValueError("queue_capacity cannot be negative")
        if completion_ttl_seconds is not None and completion_ttl_seconds <= 0:
            raise ValueError("completion_ttl_seconds must be positive or None")
        if preview_chars < 32:
            raise ValueError("preview_chars must be at least 32")
        if max_payload_chars < preview_chars:
            raise ValueError("max_payload_chars must be no smaller than preview_chars")

        self.db_path = str(Path(db_path))
        self.max_workers = max_workers
        self.queue_capacity = queue_capacity
        self.completion_ttl_seconds = completion_ttl_seconds
        self.preview_chars = preview_chars
        self.max_payload_chars = max_payload_chars
        self._state_lock = threading.RLock()
        self._closed = False
        self._futures: dict[str, Future[Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._slots = threading.BoundedSemaphore(max_workers + queue_capacity)
        self._execution_slots = threading.BoundedSemaphore(max_workers)

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.recovered_jobs = self._recover_interrupted_jobs()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent-job")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    progress REAL,
                    detail TEXT,
                    metadata_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT,
                    expires_at TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_status_created_idx
                    ON jobs(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS jobs_kind_created_idx
                    ON jobs(kind, created_at DESC);

                CREATE TABLE IF NOT EXISTS job_progress (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    progress REAL,
                    detail TEXT,
                    data_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS job_progress_job_sequence_idx
                    ON job_progress(job_id, sequence);
                """
            )

    def _recover_interrupted_jobs(self) -> int:
        """Fail work that could not have survived the previous process."""
        now = _utc_now()
        expires_at = self._expiry(now)
        message = "interrupted by process restart"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT job_id FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchall()
            for row in rows:
                db.execute(
                    """INSERT INTO job_progress(
                           job_id, status, phase, progress, detail, data_json, created_at
                       ) VALUES (?, 'failed', 'interrupted', NULL, ?, NULL, ?)""",
                    (row["job_id"], message, now),
                )
            db.execute(
                """UPDATE jobs
                   SET status='failed', phase='interrupted', detail=?, error=?,
                       result_json=NULL, updated_at=?, finished_at=?, expires_at=?
                   WHERE status IN ('queued', 'running')""",
                (message, message, now, now, expires_at),
            )
            db.commit()
        return len(rows)

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in JOB_STATUSES:
            raise ValueError(f"unsupported job status: {status}")

    def _truncate(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        if len(text) <= self.preview_chars:
            return text
        return text[: self.preview_chars - 1] + "…"

    def _serialize(self, value: Any, *, truncate: bool = True) -> str:
        try:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError, RecursionError):
            rendered = json.dumps(str(value), ensure_ascii=False)
        if not truncate or len(rendered) <= self.max_payload_chars:
            return rendered
        preview = rendered[: max(1, self.max_payload_chars - 64)] + "…"
        return json.dumps({"_truncated": True, "preview": preview}, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _deserialize(value: str | None) -> Any:
        return json.loads(value) if value is not None else None

    def _expiry(self, finished_at: str) -> str | None:
        if self.completion_ttl_seconds is None:
            return None
        finished = datetime.fromisoformat(finished_at)
        return (finished + timedelta(seconds=self.completion_ttl_seconds)).isoformat(timespec="microseconds")

    def _acquire_slot(self, block: bool, timeout: float | None) -> bool:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")
        if not block:
            return self._slots.acquire(blocking=False)
        if timeout is None:
            return self._slots.acquire()
        return self._slots.acquire(timeout=timeout)

    def submit(
        self,
        kind: str,
        worker: JobWorker,
        *,
        metadata: dict[str, Any] | None = None,
        job_id: str | None = None,
        block: bool = False,
        timeout: float | None = None,
    ) -> str:
        """Persist and schedule work, applying backpressure at capacity.

        ``worker`` receives ``update`` and ``cancelled`` callables.  The normal
        update signature is ``update(phase, percent=None, detail=None,
        data=None)``.  ``update(phase, detail)`` is also accepted for tasks
        without numeric progress.
        """
        if not kind.strip():
            raise ValueError("kind cannot be empty")
        if not callable(worker):
            raise TypeError("worker must be callable")
        with self._state_lock:
            if self._closed:
                raise ManagerClosedError("job manager has been shut down")
        self.cleanup_expired()
        if not self._acquire_slot(block, timeout):
            raise QueueFullError(
                f"job capacity exhausted ({self.max_workers} running, {self.queue_capacity} queued)"
            )

        identifier = job_id or uuid.uuid4().hex
        now = _utc_now()
        cancel_event = threading.Event()
        try:
            with self._connect() as db:
                db.execute(
                    """INSERT INTO jobs(
                           job_id, kind, status, phase, progress, detail,
                           metadata_json, created_at, updated_at
                       ) VALUES (?, ?, 'queued', 'queued', 0, ?, ?, ?, ?)""",
                    (identifier, kind, "waiting for a worker", self._serialize(metadata or {}), now, now),
                )
                self._insert_history(
                    db,
                    identifier,
                    status="queued",
                    phase="queued",
                    progress=0.0,
                    detail="waiting for a worker",
                    data=None,
                    created_at=now,
                )
            with self._state_lock:
                if self._closed:
                    self._mark_terminal(identifier, "cancelled", detail="manager shut down before scheduling")
                    raise ManagerClosedError("job manager has been shut down")
                self._cancel_events[identifier] = cancel_event
                future = self._executor.submit(
                    self._run_job_with_slot,
                    identifier,
                    worker,
                    cancel_event,
                )
                self._futures[identifier] = future
                future.add_done_callback(lambda _future, value=identifier: self._release_job_slot(value))
            return identifier
        except Exception:
            with self._state_lock:
                scheduled = identifier in self._futures
                if not scheduled:
                    self._cancel_events.pop(identifier, None)
            if not scheduled:
                self._slots.release()
            raise

    @contextmanager
    def reserve(self, *, block: bool = False, timeout: float | None = None):
        """Share this manager's capacity gate with synchronous compatibility work."""

        with self._state_lock:
            if self._closed:
                raise ManagerClosedError("job manager has been shut down")
        if not self._acquire_slot(block, timeout):
            raise QueueFullError(
                f"job capacity exhausted ({self.max_workers} running, {self.queue_capacity} queued)"
            )
        if not self._execution_slots.acquire(blocking=False):
            self._slots.release()
            raise QueueFullError(f"all {self.max_workers} execution slots are busy")
        try:
            yield
        finally:
            self._execution_slots.release()
            self._slots.release()

    def _run_job_with_slot(
        self,
        job_id: str,
        worker: JobWorker,
        cancel_event: threading.Event,
    ) -> None:
        self._execution_slots.acquire()
        try:
            self._run_job(job_id, worker, cancel_event)
        finally:
            self._execution_slots.release()

    def _release_job_slot(self, job_id: str) -> None:
        with self._state_lock:
            self._futures.pop(job_id, None)
            self._cancel_events.pop(job_id, None)
        self._slots.release()

    def _run_job(self, job_id: str, worker: JobWorker, cancel_event: threading.Event) -> None:
        if cancel_event.is_set() or self._is_cancel_requested(job_id):
            self._mark_terminal(job_id, "cancelled", detail="cancelled before execution")
            return
        if not self._mark_running(job_id):
            return

        def cancelled() -> bool:
            return cancel_event.is_set() or self._is_cancel_requested(job_id)

        def update(
            phase: str | float | int,
            percent: float | str | None = None,
            detail: str | None = None,
            data: Any = None,
        ) -> bool:
            actual_phase: str
            actual_percent: float | None
            actual_detail = detail
            if isinstance(phase, (int, float)):
                actual_phase = "progress"
                actual_percent = float(phase)
                if isinstance(percent, str) and actual_detail is None:
                    actual_detail = percent
            else:
                actual_phase = phase
                if isinstance(percent, str) and actual_detail is None:
                    actual_percent = None
                    actual_detail = percent
                elif percent is None:
                    actual_percent = None
                else:
                    actual_percent = float(percent)
            return self._record_progress(
                job_id,
                actual_phase,
                actual_percent,
                actual_detail,
                data,
                cancel_event,
            )

        try:
            result = worker(update, cancelled)
            if cancelled():
                self._mark_terminal(job_id, "cancelled", detail="cancelled by request")
            else:
                self._mark_terminal(job_id, "completed", result=result, detail="completed")
        except JobCancelled as exc:
            self._mark_terminal(job_id, "cancelled", detail=str(exc) or "cancelled by worker")
        except Exception as exc:
            self._mark_terminal(
                job_id,
                "failed",
                error=f"{type(exc).__name__}: {exc}",
                detail="worker failed",
            )

    def _mark_running(self, job_id: str) -> bool:
        now = _utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status, cancel_requested FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "queued" or row["cancel_requested"]:
                db.commit()
                if row and row["status"] == "queued" and row["cancel_requested"]:
                    self._mark_terminal(job_id, "cancelled", detail="cancelled before execution")
                return False
            db.execute(
                """UPDATE jobs
                   SET status='running', phase='running', detail=?, started_at=?, updated_at=?
                   WHERE job_id=? AND status='queued'""",
                ("worker started", now, now, job_id),
            )
            self._insert_history(
                db,
                job_id,
                status="running",
                phase="running",
                progress=0.0,
                detail="worker started",
                data=None,
                created_at=now,
            )
            db.commit()
        return True

    def _record_progress(
        self,
        job_id: str,
        phase: str,
        progress: float | None,
        detail: str | None,
        data: Any,
        cancel_event: threading.Event,
    ) -> bool:
        if not phase.strip():
            raise ValueError("phase cannot be empty")
        if progress is not None and not 0 <= progress <= 100:
            raise ValueError("progress must be between 0 and 100")
        now = _utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status, cancel_requested FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "running" or row["cancel_requested"] or cancel_event.is_set():
                db.commit()
                return False
            db.execute(
                """UPDATE jobs
                   SET phase=?, progress=COALESCE(?, progress), detail=?, updated_at=?
                   WHERE job_id=? AND status='running'""",
                (phase, progress, self._truncate(detail), now, job_id),
            )
            self._insert_history(
                db,
                job_id,
                status="running",
                phase=phase,
                progress=progress,
                detail=detail,
                data=data,
                created_at=now,
            )
            db.commit()
        return True

    def _mark_terminal(
        self,
        job_id: str,
        status: str,
        *,
        result: Any = None,
        error: str | None = None,
        detail: str | None = None,
    ) -> bool:
        if status not in TERMINAL_STATUSES:
            raise ValueError("terminal status required")
        now = _utc_now()
        expires_at = self._expiry(now)
        progress = 100.0 if status == "completed" else None
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row["status"] not in ACTIVE_STATUSES:
                db.commit()
                return False
            db.execute(
                """UPDATE jobs
                   SET status=?, phase=?, progress=COALESCE(?, progress), detail=?,
                       result_json=?, error=?, updated_at=?, finished_at=?, expires_at=?
                   WHERE job_id=? AND status IN ('queued', 'running')""",
                (
                    status,
                    status,
                    progress,
                    self._truncate(detail),
                    # A completed result is part of the public job contract.  Replacing
                    # an oversized result with a preview removes required fields such
                    # as ``answer`` and ``media`` from chat polling responses.  Keep
                    # final results intact; the payload limit still applies to
                    # metadata and progress-event data, which are diagnostic only.
                    self._serialize(result, truncate=False) if status == "completed" else None,
                    self._truncate(error),
                    now,
                    now,
                    expires_at,
                    job_id,
                ),
            )
            self._insert_history(
                db,
                job_id,
                status=status,
                phase=status,
                progress=progress,
                detail=detail,
                data=None,
                created_at=now,
            )
            db.commit()
        return True

    def _is_cancel_requested(self, job_id: str) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT cancel_requested, status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return bool(row is None or row["cancel_requested"] or row["status"] == "cancelled")

    def cancel(self, job_id: str) -> bool:
        """Request cancellation; queued futures are cancelled immediately."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status, cancel_requested FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                db.commit()
                return False
            if row["status"] == "cancelled":
                db.commit()
                return True
            if row["status"] in {"completed", "failed"}:
                db.commit()
                return False
            now = _utc_now()
            db.execute(
                "UPDATE jobs SET cancel_requested=1, updated_at=? WHERE job_id=? AND status IN ('queued', 'running')",
                (now, job_id),
            )
            self._insert_history(
                db,
                job_id,
                status=row["status"],
                phase="cancel_requested",
                progress=None,
                detail="cooperative cancellation requested",
                data=None,
                created_at=now,
            )
            db.commit()

        with self._state_lock:
            event = self._cancel_events.get(job_id)
            future = self._futures.get(job_id)
            if event:
                event.set()
        if row["status"] == "queued" and future and future.cancel():
            self._mark_terminal(job_id, "cancelled", detail="cancelled while queued")
        return True

    def _insert_history(
        self,
        db: sqlite3.Connection,
        job_id: str,
        *,
        status: str,
        phase: str,
        progress: float | None,
        detail: str | None,
        data: Any,
        created_at: str,
    ) -> None:
        data_json = self._serialize(data) if data is not None else None
        db.execute(
            """INSERT INTO job_progress(
                   job_id, status, phase, progress, detail, data_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (job_id, status, phase, progress, self._truncate(detail), data_json, created_at),
        )

    def get(self, job_id: str, *, include_history: bool = True) -> dict[str, Any] | None:
        self.cleanup_expired()
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            record = self._job_record(row)
            if include_history:
                history = db.execute(
                    "SELECT * FROM job_progress WHERE job_id=? ORDER BY sequence",
                    (job_id,),
                ).fetchall()
                record["history"] = [self._history_record(item) for item in history]
        return record

    get_job = get

    def list(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self.cleanup_expired()
        if status is not None:
            self._validate_status(status)
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        conditions: list[str] = []
        parameters: list[Any] = []
        if status:
            conditions.append("status=?")
            parameters.append(status)
        if kind:
            conditions.append("kind=?")
            parameters.append(kind)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM jobs{where} ORDER BY created_at DESC, job_id DESC LIMIT ? OFFSET ?"
        parameters.extend((limit, offset))
        with self._connect() as db:
            rows = db.execute(query, parameters).fetchall()
        return [self._job_record(row) for row in rows]

    list_jobs = list

    def cleanup_expired(self, *, now: datetime | str | None = None) -> int:
        threshold = _as_utc(now).isoformat(timespec="microseconds")
        with self._connect() as db:
            cursor = db.execute(
                """DELETE FROM jobs
                   WHERE status IN ('completed', 'failed', 'cancelled')
                     AND expires_at IS NOT NULL AND expires_at<=?""",
                (threshold,),
            )
        return cursor.rowcount

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_pending: bool = False,
        cancel_running: bool = False,
    ) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            futures = dict(self._futures)
            events = dict(self._cancel_events)
        if cancel_pending or cancel_running:
            for job_id, future in futures.items():
                if cancel_running or (cancel_pending and not future.running()):
                    event = events.get(job_id)
                    if event:
                        event.set()
                    self.cancel(job_id)
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)

    def __enter__(self) -> PersistentJobManager:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.shutdown(wait=True, cancel_pending=True, cancel_running=True)

    @staticmethod
    def _job_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "status": row["status"],
            "phase": row["phase"],
            "progress": row["progress"],
            "detail": row["detail"],
            "metadata": json.loads(row["metadata_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] is not None else None,
            "error": row["error"],
            "cancel_requested": bool(row["cancel_requested"]),
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "finished_at": row["finished_at"],
            "expires_at": row["expires_at"],
        }

    @staticmethod
    def _history_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "sequence": row["sequence"],
            "status": row["status"],
            "phase": row["phase"],
            "progress": row["progress"],
            "detail": row["detail"],
            "data": json.loads(row["data_json"]) if row["data_json"] is not None else None,
            "created_at": row["created_at"],
        }


JobManager = PersistentJobManager
