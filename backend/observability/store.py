"""SQLite-backed run, span, tool-event, and review-checkpoint storage.

The store intentionally has no dependency on FastAPI or LangGraph.  Agent
runtimes can therefore emit traces before either framework is initialized and
can recover human-review state after a process restart.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_STATUSES = frozenset({"running", "awaiting_review", "completed", "failed"})
EVENT_STATUSES = frozenset({"running", "completed", "failed"})
REDACTED = "[REDACTED]"


class RunNotFoundError(KeyError):
    """Raised when a requested run does not exist."""


class InvalidTransitionError(RuntimeError):
    """Raised when a run cannot move from its current state to the target."""


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "passwd",
        "password",
        "private_key",
        "refresh_token",
        "access_token",
        "secret",
        "set_cookie",
        "token",
    }
)
_SENSITIVE_COMPACT_KEYS = frozenset(key.replace("_", "") for key in _SENSITIVE_KEYS)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_KEY_ASSIGNMENT_PATTERN = re.compile(
    r"(?ix)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret|authorization)"
    r"(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)"
)
_OPENAI_STYLE_KEY_PATTERN = re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _elapsed_ms(started_at: str, ended_at: str) -> float:
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(ended_at)
    return max(0.0, (end - start).total_seconds() * 1000.0)


def _is_sensitive_key(key: object) -> bool:
    name = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    compact = name.replace("_", "")
    if name in _SENSITIVE_KEYS or compact in _SENSITIVE_COMPACT_KEYS:
        return True
    return any(
        name.endswith(suffix)
        for suffix in ("_api_key", "_password", "_passwd", "_secret", "_credential", "_access_token", "_refresh_token")
    )


def _redact_text(value: str) -> str:
    value = _BEARER_PATTERN.sub("Bearer [REDACTED]", value)
    value = _KEY_ASSIGNMENT_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", value)
    return _OPENAI_STYLE_KEY_PATTERN.sub(REDACTED, value)


def _sanitize(value: Any) -> Any:
    """Return a JSON-compatible copy with secrets removed recursively."""
    if isinstance(value, dict):
        return {
            str(key): REDACTED if _is_sensitive_key(key) else _sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value))


class ObservabilityStore:
    """Durable, thread-safe storage for one or more Agent runtimes."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        preview_chars: int = 512,
        max_payload_chars: int = 16_384,
        max_checkpoint_chars: int = 1_000_000,
    ) -> None:
        if preview_chars < 8:
            raise ValueError("preview_chars must be at least 8")
        if max_payload_chars < preview_chars:
            raise ValueError("max_payload_chars must be no smaller than preview_chars")
        if max_checkpoint_chars < max_payload_chars:
            raise ValueError("max_checkpoint_chars must be no smaller than max_payload_chars")
        self.db_path = str(Path(db_path))
        self.preview_chars = preview_chars
        self.max_payload_chars = max_payload_chars
        self.max_checkpoint_chars = max_checkpoint_chars
        self._schema_lock = threading.RLock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._schema_lock, self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    question_preview TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    result_preview TEXT,
                    error_preview TEXT,
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    duration_ms REAL
                );
                CREATE INDEX IF NOT EXISTS runs_status_started_idx
                    ON runs(status, started_at DESC);
                CREATE INDEX IF NOT EXISTS runs_thread_started_idx
                    ON runs(thread_id, started_at DESC);

                CREATE TABLE IF NOT EXISTS spans (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    span_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    parent_span_id TEXT,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    error_preview TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_ms REAL
                );
                CREATE INDEX IF NOT EXISTS spans_run_sequence_idx
                    ON spans(run_id, sequence);

                CREATE TABLE IF NOT EXISTS tool_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    span_id TEXT,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    output_preview TEXT,
                    error_preview TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    duration_ms REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS tool_events_run_sequence_idx
                    ON tool_events(run_id, sequence);

                CREATE TABLE IF NOT EXISTS review_checkpoints (
                    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
                    version INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL,
                    reason_preview TEXT,
                    approved INTEGER,
                    feedback_preview TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resumed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS delivery_outbox (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error_preview TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    delivered_at TEXT
                );
                CREATE INDEX IF NOT EXISTS delivery_outbox_state_created_idx
                    ON delivery_outbox(state, created_at);
                CREATE INDEX IF NOT EXISTS delivery_outbox_run_idx
                    ON delivery_outbox(run_id, created_at);
                """
            )

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    def _preview(self, value: Any) -> str | None:
        if value is None:
            return None
        clean = _sanitize(value)
        if isinstance(clean, str):
            rendered = clean
        else:
            rendered = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return self._truncate(rendered, self.preview_chars)

    def _json_payload(self, value: Any) -> str:
        clean = _sanitize({} if value is None else value)
        rendered = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(rendered) <= self.max_payload_chars:
            return rendered
        envelope = {
            "_truncated": True,
            "preview": self._truncate(rendered, max(self.preview_chars, self.max_payload_chars - 64)),
        }
        return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _durable_payload(self, value: Any) -> str:
        """Serialize resumable state losslessly, with an explicit hard limit."""

        clean = _sanitize({} if value is None else value)
        rendered = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(rendered) > self.max_checkpoint_chars:
            raise ValueError(
                f"durable payload exceeds max_checkpoint_chars={self.max_checkpoint_chars}"
            )
        return rendered

    @staticmethod
    def _validate_status(status: str, allowed: frozenset[str]) -> None:
        if status not in allowed:
            raise ValueError(f"Unsupported status: {status}")

    def _require_run(self, db: sqlite3.Connection, run_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return row

    def create_run(
        self,
        thread_id: str,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> str:
        """Create a running trace and return its stable run id."""
        if not thread_id.strip():
            raise ValueError("thread_id cannot be empty")
        identifier = run_id or uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO runs(
                       run_id, thread_id, status, question_preview, metadata_json,
                       started_at, updated_at
                   ) VALUES (?, ?, 'running', ?, ?, ?, ?)""",
                (identifier, thread_id, self._preview(question) or "", self._json_payload(metadata), now, now),
            )
        return identifier

    def start_span(
        self,
        run_id: str,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        parent_span_id: str | None = None,
        span_id: str | None = None,
    ) -> str:
        """Open a span. ``finish_span`` is idempotent for an already-ended span."""
        identifier = span_id or uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as db:
            self._require_run(db, run_id)
            db.execute(
                """INSERT INTO spans(
                       span_id, run_id, parent_span_id, name, status,
                       attributes_json, started_at
                   ) VALUES (?, ?, ?, ?, 'running', ?, ?)""",
                (identifier, run_id, parent_span_id, name, self._json_payload(attributes), now),
            )
        return identifier

    def finish_span(
        self,
        span_id: str,
        *,
        status: str = "completed",
        attributes: dict[str, Any] | None = None,
        error: Any = None,
    ) -> dict[str, Any]:
        self._validate_status(status, EVENT_STATUSES - {"running"})
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM spans WHERE span_id=?", (span_id,)).fetchone()
            if row is None:
                raise KeyError(span_id)
            if row["status"] != "running":
                db.commit()
                return self._span_record(row)
            current_attributes = json.loads(row["attributes_json"])
            if attributes:
                if not isinstance(current_attributes, dict):
                    current_attributes = {"initial": current_attributes}
                current_attributes.update(_sanitize(attributes))
            ended_at = _utc_now()
            db.execute(
                """UPDATE spans
                   SET status=?, attributes_json=?, error_preview=?, ended_at=?, duration_ms=?
                   WHERE span_id=?""",
                (
                    status,
                    self._json_payload(current_attributes),
                    self._preview(error),
                    ended_at,
                    _elapsed_ms(row["started_at"], ended_at),
                    span_id,
                ),
            )
            updated = db.execute("SELECT * FROM spans WHERE span_id=?", (span_id,)).fetchone()
            db.commit()
        return self._span_record(updated)

    def record_tool_event(
        self,
        run_id: str,
        tool_name: str,
        *,
        arguments: Any = None,
        output: Any = None,
        status: str = "completed",
        error: Any = None,
        duration_ms: float = 0.0,
        span_id: str | None = None,
        event_id: str | None = None,
        started_at: str | None = None,
    ) -> str:
        """Append an immutable tool invocation event."""
        self._validate_status(status, EVENT_STATUSES - {"running"})
        if duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")
        identifier = event_id or uuid.uuid4().hex
        started = started_at or _utc_now()
        ended = _utc_now()
        with self._connect() as db:
            self._require_run(db, run_id)
            db.execute(
                """INSERT INTO tool_events(
                       event_id, run_id, span_id, tool_name, status,
                       arguments_json, output_preview, error_preview,
                       started_at, ended_at, duration_ms
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    identifier,
                    run_id,
                    span_id,
                    tool_name,
                    status,
                    self._json_payload(arguments),
                    self._preview(output),
                    self._preview(error),
                    started,
                    ended,
                    float(duration_ms),
                ),
            )
        return identifier

    def complete_run(
        self,
        run_id: str,
        *,
        result: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._finish_run(run_id, "completed", result=result, metadata=metadata)

    def fail_run(
        self,
        run_id: str,
        error: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._finish_run(run_id, "failed", error=error, metadata=metadata)

    def _finish_run(
        self,
        run_id: str,
        status: str,
        *,
        result: Any = None,
        error: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_status(status, frozenset({"completed", "failed"}))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._require_run(db, run_id)
            if row["status"] == status:
                db.commit()
                return self.get_run(run_id, include_events=False)  # type: ignore[return-value]
            if row["status"] in {"completed", "failed"}:
                raise InvalidTransitionError(f"run {run_id} is already {row['status']}")
            if row["status"] == "awaiting_review" and status == "completed":
                raise InvalidTransitionError("resume the review checkpoint before completing the run")
            existing_metadata = json.loads(row["metadata_json"])
            if metadata:
                if not isinstance(existing_metadata, dict):
                    existing_metadata = {"initial": existing_metadata}
                existing_metadata.update(_sanitize(metadata))
            ended_at = _utc_now()
            db.execute(
                """UPDATE runs
                   SET status=?, metadata_json=?, result_preview=?, error_preview=?,
                       updated_at=?, completed_at=?, duration_ms=?
                   WHERE run_id=?""",
                (
                    status,
                    self._json_payload(existing_metadata),
                    self._preview(result),
                    self._preview(error),
                    ended_at,
                    ended_at,
                    _elapsed_ms(row["started_at"], ended_at),
                    run_id,
                ),
            )
            db.commit()
        return self.get_run(run_id, include_events=False)  # type: ignore[return-value]

    def save_review_checkpoint(
        self,
        run_id: str,
        checkpoint: dict[str, Any],
        *,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """Atomically pause a running run and persist resumable state.

        Repeating the call while the same run is already waiting is a no-op.
        After a successful resume, the run may later create a new checkpoint;
        its monotonically increasing ``version`` distinguishes the cycles.
        """
        checkpoint_json = self._durable_payload(checkpoint)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            run = self._require_run(db, run_id)
            existing = db.execute("SELECT * FROM review_checkpoints WHERE run_id=?", (run_id,)).fetchone()
            if run["status"] == "awaiting_review" and existing and existing["state"] == "pending":
                db.commit()
                return self._review_record(existing)
            if run["status"] != "running":
                raise InvalidTransitionError(f"cannot checkpoint run in {run['status']} state")
            now = _utc_now()
            version = int(existing["version"]) + 1 if existing else 1
            db.execute(
                """INSERT INTO review_checkpoints(
                       run_id, version, state, checkpoint_json, reason_preview,
                       approved, feedback_preview, created_at, updated_at, resumed_at
                   ) VALUES (?, ?, 'pending', ?, ?, NULL, NULL, ?, ?, NULL)
                   ON CONFLICT(run_id) DO UPDATE SET
                       version=excluded.version,
                       state='pending',
                       checkpoint_json=excluded.checkpoint_json,
                       reason_preview=excluded.reason_preview,
                       approved=NULL,
                       feedback_preview=NULL,
                       created_at=excluded.created_at,
                       updated_at=excluded.updated_at,
                       resumed_at=NULL""",
                (run_id, version, checkpoint_json, self._preview(reason), now, now),
            )
            db.execute(
                "UPDATE runs SET status='awaiting_review', updated_at=? WHERE run_id=?",
                (now, run_id),
            )
            saved = db.execute("SELECT * FROM review_checkpoints WHERE run_id=?", (run_id,)).fetchone()
            db.commit()
        return self._review_record(saved)

    mark_awaiting_review = save_review_checkpoint

    def resume_review(
        self,
        run_id: str,
        approved: bool,
        *,
        feedback: str | None = None,
        next_status: str = "running",
        outbox_event: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resume a review once; duplicate requests return the first decision.

        The compare-and-set transition and checkpoint decision are committed in
        one ``BEGIN IMMEDIATE`` transaction.  Competing callers therefore see
        exactly one response with ``resumed=True``.
        """
        self._validate_status(next_status, frozenset({"running", "completed", "failed"}))
        outbox_json = self._durable_payload(outbox_event) if outbox_event is not None else None
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            run = self._require_run(db, run_id)
            review = db.execute("SELECT * FROM review_checkpoints WHERE run_id=?", (run_id,)).fetchone()
            if review is None:
                raise InvalidTransitionError(f"run {run_id} has no review checkpoint")
            if review["state"] == "resumed":
                outbox_id = None
                if bool(review["approved"]) and outbox_json is not None:
                    outbox_id = self._enqueue_outbox(
                        db,
                        run_id,
                        int(review["version"]),
                        outbox_json,
                    )
                db.commit()
                record = self._review_record(review)
                record.update(
                    resumed=False,
                    requested_approved=bool(approved),
                    requested_matches=record["approved"] is bool(approved),
                    run_status=run["status"],
                    outbox_event_id=outbox_id,
                )
                return record
            if run["status"] != "awaiting_review":
                raise InvalidTransitionError(f"cannot resume run in {run['status']} state")

            now = _utc_now()
            completed_at = now if next_status in {"completed", "failed"} else None
            duration_ms = _elapsed_ms(run["started_at"], now) if completed_at else None
            db.execute(
                """UPDATE review_checkpoints
                   SET state='resumed', approved=?, feedback_preview=?,
                       updated_at=?, resumed_at=?
                   WHERE run_id=? AND state='pending'""",
                (int(bool(approved)), self._preview(feedback), now, now, run_id),
            )
            db.execute(
                """UPDATE runs
                   SET status=?, updated_at=?, completed_at=?, duration_ms=?,
                       error_preview=CASE WHEN ?='failed' THEN ? ELSE error_preview END
                   WHERE run_id=? AND status='awaiting_review'""",
                (next_status, now, completed_at, duration_ms, next_status, self._preview(feedback), run_id),
            )
            outbox_id = None
            if approved and outbox_json is not None:
                outbox_id = self._enqueue_outbox(
                    db,
                    run_id,
                    int(review["version"]),
                    outbox_json,
                )
            saved = db.execute("SELECT * FROM review_checkpoints WHERE run_id=?", (run_id,)).fetchone()
            db.commit()
        record = self._review_record(saved)
        record.update(
            resumed=True,
            requested_approved=bool(approved),
            requested_matches=True,
            run_status=next_status,
            outbox_event_id=outbox_id,
        )
        return record

    @staticmethod
    def _enqueue_outbox(
        db: sqlite3.Connection,
        run_id: str,
        version: int,
        payload_json: str,
    ) -> str:
        event_id = f"review:{run_id}:{version}"
        now = _utc_now()
        db.execute(
            """INSERT OR IGNORE INTO delivery_outbox(
                   event_id, run_id, event_type, payload_json, state,
                   attempts, created_at, updated_at
               ) VALUES (?, ?, 'assistant_message', ?, 'pending', 0, ?, ?)""",
            (event_id, run_id, payload_json, now, now),
        )
        return event_id

    def get_review_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM review_checkpoints WHERE run_id=?", (run_id,)).fetchone()
        return self._review_record(row) if row else None

    def list_outbox(
        self,
        *,
        state: str = "pending",
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List durable side effects awaiting idempotent delivery."""

        if state not in {"pending", "delivered"}:
            raise ValueError("unsupported outbox state")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        conditions = ["state=?"]
        parameters: list[Any] = [state]
        if run_id:
            conditions.append("run_id=?")
            parameters.append(run_id)
        parameters.append(limit)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM delivery_outbox WHERE "
                + " AND ".join(conditions)
                + " ORDER BY created_at, event_id LIMIT ?",
                parameters,
            ).fetchall()
        return [self._outbox_record(row) for row in rows]

    def mark_outbox_delivered(self, event_id: str) -> bool:
        """Acknowledge delivery; duplicate acknowledgements are harmless."""

        now = _utc_now()
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE delivery_outbox
                   SET state='delivered', attempts=attempts+1, updated_at=?, delivered_at=?
                   WHERE event_id=? AND state='pending'""",
                (now, now, event_id),
            )
            if cursor.rowcount:
                return True
            row = db.execute(
                "SELECT state FROM delivery_outbox WHERE event_id=?",
                (event_id,),
            ).fetchone()
        return bool(row and row["state"] == "delivered")

    def mark_outbox_failed(self, event_id: str, error: Any) -> bool:
        """Record a failed attempt while retaining the event for retry."""

        now = _utc_now()
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE delivery_outbox
                   SET attempts=attempts+1, last_error_preview=?, updated_at=?
                   WHERE event_id=? AND state='pending'""",
                (self._preview(error), now, event_id),
            )
        return cursor.rowcount > 0

    def get_run(self, run_id: str, *, include_events: bool = True) -> dict[str, Any] | None:
        with self._connect() as db:
            run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None:
                return None
            record = self._run_record(run)
            if include_events:
                spans = db.execute("SELECT * FROM spans WHERE run_id=? ORDER BY sequence", (run_id,)).fetchall()
                tools = db.execute("SELECT * FROM tool_events WHERE run_id=? ORDER BY sequence", (run_id,)).fetchall()
                review = db.execute("SELECT * FROM review_checkpoints WHERE run_id=?", (run_id,)).fetchone()
                outbox = db.execute(
                    "SELECT * FROM delivery_outbox WHERE run_id=? ORDER BY created_at, event_id",
                    (run_id,),
                ).fetchall()
                record["spans"] = [self._span_record(row) for row in spans]
                record["tool_events"] = [self._tool_record(row) for row in tools]
                record["review_checkpoint"] = self._review_record(review) if review else None
                record["outbox_events"] = [self._outbox_record(row) for row in outbox]
        return record

    def list_runs(
        self,
        *,
        status: str | None = None,
        thread_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if status is not None:
            self._validate_status(status, RUN_STATUSES)
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        conditions: list[str] = []
        parameters: list[Any] = []
        if status:
            conditions.append("status=?")
            parameters.append(status)
        if thread_id:
            conditions.append("thread_id=?")
            parameters.append(thread_id)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"SELECT * FROM runs{where} ORDER BY started_at DESC, run_id DESC LIMIT ? OFFSET ?"
        parameters.extend((limit, offset))
        with self._connect() as db:
            rows = db.execute(query, parameters).fetchall()
        return [self._run_record(row) for row in rows]

    def metrics(self, *, since: str | None = None) -> dict[str, Any]:
        """Return inexpensive operational aggregates, optionally after an ISO time."""
        run_where = " WHERE started_at>=?" if since else ""
        event_where = " WHERE started_at>=?" if since else ""
        review_where = " WHERE created_at>=?" if since else ""
        parameters = (since,) if since else ()
        with self._connect() as db:
            run_rows = db.execute(
                f"SELECT status, COUNT(*) AS count FROM runs{run_where} GROUP BY status",
                parameters,
            ).fetchall()
            duration = db.execute(
                f"SELECT AVG(duration_ms) AS average FROM runs{run_where}",
                parameters,
            ).fetchone()["average"]
            run_durations = [
                float(row["duration_ms"])
                for row in db.execute(
                    f"SELECT duration_ms FROM runs{run_where} AND duration_ms IS NOT NULL"
                    if run_where else "SELECT duration_ms FROM runs WHERE duration_ms IS NOT NULL",
                    parameters,
                ).fetchall()
            ]
            span_rows = db.execute(
                f"SELECT status, COUNT(*) AS count FROM spans{event_where} GROUP BY status",
                parameters,
            ).fetchall()
            tool_rows = db.execute(
                f"SELECT status, COUNT(*) AS count FROM tool_events{event_where} GROUP BY status",
                parameters,
            ).fetchall()
            tool_duration = db.execute(
                f"SELECT AVG(duration_ms) AS average FROM tool_events{event_where}",
                parameters,
            ).fetchone()["average"]
            tool_durations = [
                float(row["duration_ms"])
                for row in db.execute(
                    f"SELECT duration_ms FROM tool_events{event_where} AND duration_ms IS NOT NULL"
                    if event_where else "SELECT duration_ms FROM tool_events WHERE duration_ms IS NOT NULL",
                    parameters,
                ).fetchall()
            ]
            review_rows = db.execute(
                f"SELECT state, approved, created_at, resumed_at FROM review_checkpoints{review_where}",
                parameters,
            ).fetchall()
            completed_without_review = db.execute(
                """SELECT COUNT(*) AS count
                   FROM runs AS run
                   LEFT JOIN review_checkpoints AS review ON review.run_id=run.run_id
                   WHERE run.status='completed' AND review.run_id IS NULL"""
                + (" AND run.started_at>=?" if since else ""),
                parameters,
            ).fetchone()["count"]
            outbox_rows = db.execute(
                "SELECT state, COUNT(*) AS count FROM delivery_outbox GROUP BY state"
            ).fetchall()

        run_counts = {status: 0 for status in sorted(RUN_STATUSES)}
        run_counts.update({row["status"]: row["count"] for row in run_rows})
        span_counts = {status: 0 for status in sorted(EVENT_STATUSES)}
        span_counts.update({row["status"]: row["count"] for row in span_rows})
        tool_counts = {status: 0 for status in sorted(EVENT_STATUSES - {"running"})}
        tool_counts.update({row["status"]: row["count"] for row in tool_rows})
        total_runs = sum(run_counts.values())
        terminal_runs = run_counts["completed"] + run_counts["failed"]
        total_spans = sum(span_counts.values())
        total_tools = sum(tool_counts.values())
        failure_rate = run_counts["failed"] / terminal_runs if terminal_runs else 0.0
        tool_failure_rate = tool_counts["failed"] / total_tools if total_tools else 0.0
        total_reviews = len(review_rows)
        pending_reviews = sum(row["state"] == "pending" for row in review_rows)
        resolved_reviews = total_reviews - pending_reviews
        approved_reviews = sum(row["state"] == "resumed" and bool(row["approved"]) for row in review_rows)
        rejected_reviews = sum(row["state"] == "resumed" and not bool(row["approved"]) for row in review_rows)
        review_waits = [
            _elapsed_ms(row["created_at"], row["resumed_at"])
            for row in review_rows
            if row["resumed_at"]
        ]
        answer_acceptance_rate = (
            (completed_without_review + approved_reviews) / terminal_runs
            if terminal_runs
            else 0.0
        )
        outbox_counts = {"pending": 0, "delivered": 0}
        outbox_counts.update({row["state"]: row["count"] for row in outbox_rows})
        return {
            "total_runs": total_runs,
            "total_spans": total_spans,
            "total_tool_events": total_tools,
            "runs": {
                "total": total_runs,
                "by_status": run_counts,
                "average_duration_ms": float(duration or 0.0),
                "p95_duration_ms": self._percentile(run_durations, 0.95),
                "failure_rate": failure_rate,
                "success_rate": (run_counts["completed"] / terminal_runs if terminal_runs else 0.0),
                "workflow_success_rate": (
                    run_counts["completed"] / terminal_runs if terminal_runs else 0.0
                ),
                "answer_acceptance_rate": answer_acceptance_rate,
                "review_rate": (total_reviews / total_runs if total_runs else 0.0),
                "pending_review_rate": (
                    run_counts["awaiting_review"] / total_runs if total_runs else 0.0
                ),
            },
            "spans": {"total": total_spans, "by_status": span_counts},
            "tools": {
                "total": total_tools,
                "by_status": tool_counts,
                "average_duration_ms": float(tool_duration or 0.0),
                "p95_duration_ms": self._percentile(tool_durations, 0.95),
                "failure_rate": tool_failure_rate,
            },
            "reviews": {
                "total": total_reviews,
                "pending": pending_reviews,
                "resolved": resolved_reviews,
                "approved": approved_reviews,
                "rejected": rejected_reviews,
                "approval_rate": (
                    approved_reviews / resolved_reviews if resolved_reviews else 0.0
                ),
                "average_wait_ms": (
                    sum(review_waits) / len(review_waits) if review_waits else 0.0
                ),
                "p95_wait_ms": self._percentile(review_waits, 0.95),
            },
            "outbox": {
                "total": sum(outbox_counts.values()),
                "by_state": outbox_counts,
                "pending": outbox_counts["pending"],
            },
        }

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        """Return a nearest-rank percentile without a statistics dependency."""

        if not values:
            return 0.0
        ordered = sorted(values)
        rank = max(1, min(len(ordered), int(len(ordered) * quantile + 0.999999)))
        return float(ordered[rank - 1])

    @staticmethod
    def _run_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "thread_id": row["thread_id"],
            "status": row["status"],
            "question_preview": row["question_preview"],
            "metadata": json.loads(row["metadata_json"]),
            "result_preview": row["result_preview"],
            "error_preview": row["error_preview"],
            "started_at": row["started_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "duration_ms": row["duration_ms"],
        }

    @staticmethod
    def _span_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "span_id": row["span_id"],
            "run_id": row["run_id"],
            "parent_span_id": row["parent_span_id"],
            "name": row["name"],
            "status": row["status"],
            "attributes": json.loads(row["attributes_json"]),
            "error_preview": row["error_preview"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "duration_ms": row["duration_ms"],
        }

    @staticmethod
    def _tool_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "run_id": row["run_id"],
            "span_id": row["span_id"],
            "tool_name": row["tool_name"],
            "status": row["status"],
            "arguments": json.loads(row["arguments_json"]),
            "output_preview": row["output_preview"],
            "error_preview": row["error_preview"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "duration_ms": row["duration_ms"],
        }

    @staticmethod
    def _review_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "version": row["version"],
            "state": row["state"],
            "checkpoint": json.loads(row["checkpoint_json"]),
            "reason_preview": row["reason_preview"],
            "approved": None if row["approved"] is None else bool(row["approved"]),
            "feedback_preview": row["feedback_preview"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "resumed_at": row["resumed_at"],
        }

    @staticmethod
    def _outbox_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "run_id": row["run_id"],
            "event_type": row["event_type"],
            "payload": json.loads(row["payload_json"]),
            "state": row["state"],
            "attempts": row["attempts"],
            "last_error_preview": row["last_error_preview"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "delivered_at": row["delivered_at"],
        }
