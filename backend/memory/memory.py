"""Thread-safe short- and long-term SQLite memory."""
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY, thread_id TEXT, role TEXT, content TEXT, "
                "created_at TEXT, idempotency_key TEXT)"
            )
            message_columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)").fetchall()}
            if "idempotency_key" not in message_columns:
                db.execute("ALTER TABLE messages ADD COLUMN idempotency_key TEXT")
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS messages_idempotency_key_idx "
                "ON messages(idempotency_key) WHERE idempotency_key IS NOT NULL"
            )
            db.execute("CREATE TABLE IF NOT EXISTS memories (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS reviews (thread_id TEXT PRIMARY KEY, payload TEXT, status TEXT, updated_at TEXT)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS episodes ("
                "id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL, content TEXT NOT NULL, "
                "metadata TEXT NOT NULL, importance REAL NOT NULL, expires_at TEXT, created_at TEXT NOT NULL)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS messages_thread_id_idx ON messages(thread_id, id DESC)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS episodes_thread_created_idx ON episodes(thread_id, created_at DESC)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        idempotency_key: str | None = None,
    ) -> bool:
        """Append a message, or acknowledge an already persisted keyed write."""

        with self._connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO messages(thread_id, role, content, created_at, idempotency_key) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, role, content, self._now(), idempotency_key),
            )
        return cursor.rowcount > 0

    def history(self, thread_id: str, limit: int = 10) -> list[dict[str, str]]:
        with self._connect() as db:
            rows = db.execute("SELECT role, content FROM messages WHERE thread_id=? ORDER BY id DESC LIMIT ?", (thread_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def remember(self, key: str, value: object) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO memories VALUES (?, ?, ?)", (key, json.dumps(value, ensure_ascii=False), self._now()))

    def recall(self, key: str, default: object = None) -> object:
        with self._connect() as db:
            row = db.execute("SELECT value FROM memories WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def save_review(self, thread_id: str, payload: dict) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO reviews VALUES (?, ?, 'pending', ?)", (thread_id, json.dumps(payload, ensure_ascii=False), self._now()))

    def resolve_review(self, thread_id: str, approved: bool) -> bool:
        with self._connect() as db:
            cursor = db.execute("UPDATE reviews SET status=?, updated_at=? WHERE thread_id=? AND status='pending'", ("approved" if approved else "rejected", self._now(), thread_id))
        return cursor.rowcount > 0

    def remember_episode(
        self,
        thread_id: str,
        content: str,
        metadata: dict | None = None,
        importance: float = 0.5,
        ttl_days: int | None = 30,
    ) -> int:
        """Store a scoped episodic memory with importance and optional expiry."""

        expires_at = None
        if ttl_days is not None:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=max(1, ttl_days))).isoformat()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO episodes(thread_id, content, metadata, importance, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    max(0.0, min(1.0, float(importance))),
                    expires_at,
                    self._now(),
                ),
            )
            return int(cursor.lastrowid)

    def search_episodes(self, thread_id: str, query: str, limit: int = 5) -> list[dict]:
        """Retrieve relevant non-expired memories using transparent lexical scoring."""

        now = self._now()
        with self._connect() as db:
            db.execute("DELETE FROM episodes WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
            rows = db.execute(
                "SELECT id, content, metadata, importance, expires_at, created_at FROM episodes "
                "WHERE thread_id=? ORDER BY created_at DESC LIMIT 200",
                (thread_id,),
            ).fetchall()
        query_tokens = self._memory_tokens(query)
        ranked = []
        for row in rows:
            content_tokens = self._memory_tokens(row["content"])
            overlap = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            score = 0.75 * overlap + 0.25 * float(row["importance"])
            if query_tokens and overlap == 0:
                continue
            ranked.append((score, row))
        ranked.sort(key=lambda item: (item[0], item[1]["created_at"]), reverse=True)
        return [
            {
                "id": int(row["id"]),
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "importance": float(row["importance"]),
                "expires_at": row["expires_at"],
                "score": round(score, 6),
            }
            for score, row in ranked[:max(1, min(int(limit), 20))]
        ]

    @staticmethod
    def _memory_tokens(text: str) -> set[str]:
        lowered = text.lower()
        words = set(re.findall(r"[a-z0-9_\-]{2,}", lowered))
        chinese = re.findall(r"[\u4e00-\u9fff]+", lowered)
        words.update(run[index:index + 2] for run in chinese for index in range(max(1, len(run) - 1)))
        return {token for token in words if token}
