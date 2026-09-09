"""Persistent SQLite vector store with cosine search."""
import json
import math
import sqlite3
import threading
from pathlib import Path
from collections.abc import Callable
from backend.embedding.base import EmbeddingModel
from backend.models import Chunk, SearchResult

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by the dependency-light CI job
    np = None

class VectorStore:
    """Small local vector store; Chroma-compatible service boundary."""
    def __init__(self, db_path: str | Path, embedding: EmbeddingModel) -> None:
        self.db_path = str(db_path)
        self.embedding = embedding
        self._snapshot_lock = threading.RLock()
        self._cached_chunks: tuple[Chunk, ...] | None = None
        self._cached_vectors = None
        self._cached_norms = None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=NORMAL")
            db.execute("CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL, embedding TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS store_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _embedding_signature(self) -> str:
        """Return a stable identity for vectors produced by the active model."""
        model = self.embedding
        details = {
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "dimensions": getattr(model, "dimensions", None),
            "model_name": getattr(model, "model_name", None),
        }
        return json.dumps(details, ensure_ascii=False, sort_keys=True)

    def _embed(self, chunks: list[Chunk], progress: Callable[[int, int], None] | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors.extend(self.embedding.embed_documents([chunk.text for chunk in batch]))
            if progress:
                progress(min(start + len(batch), len(chunks)), len(chunks))
        return vectors

    def replace(self, chunks: list[Chunk], progress: Callable[[int, int], None] | None = None, storing: Callable[[], None] | None = None) -> None:
        vectors = self._embed(chunks, progress)
        if storing:
            storing()
        with self._connect() as db:
            db.execute("DELETE FROM chunks")
            db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", [(c.chunk_id, c.text, json.dumps(c.metadata(), ensure_ascii=False), json.dumps(v)) for c, v in zip(chunks, vectors)])
            db.execute("INSERT OR REPLACE INTO store_metadata VALUES ('embedding_signature', ?)", (self._embedding_signature(),))
        self._invalidate_snapshot()

    def sync(
        self,
        chunks: list[Chunk],
        progress: Callable[[int, int], None] | None = None,
        storing: Callable[[], None] | None = None,
        preserve_source_paths: set[str] | None = None,
    ) -> dict[str, int]:
        """Synchronize chunks, embedding only new or changed content."""
        incoming = {chunk.chunk_id: chunk for chunk in chunks}
        preserved_sources = preserve_source_paths or set()
        with self._connect() as db:
            rows = db.execute("SELECT chunk_id, text, metadata FROM chunks").fetchall()
            signature_row = db.execute("SELECT value FROM store_metadata WHERE key='embedding_signature'").fetchone()

        existing_ids = {row["chunk_id"] for row in rows}
        sources_by_id = {row["chunk_id"]: json.loads(row["metadata"]).get("source_path", "") for row in rows}
        deleted_ids = {
            chunk_id for chunk_id in existing_ids - incoming.keys()
            if sources_by_id.get(chunk_id) not in preserved_sources
        }
        signature_matches = bool(signature_row and signature_row["value"] == self._embedding_signature())
        unchanged_ids = existing_ids & incoming.keys() if signature_matches else set()
        pending = [chunk for chunk_id, chunk in incoming.items() if chunk_id not in unchanged_ids]
        vectors = self._embed(pending, progress)
        if storing:
            storing()

        with self._connect() as db:
            if deleted_ids:
                db.executemany("DELETE FROM chunks WHERE chunk_id=?", [(chunk_id,) for chunk_id in deleted_ids])
            # Metadata can change without changing text (for example a PDF title),
            # so refresh it for retained chunks without recomputing their vectors.
            db.executemany(
                "UPDATE chunks SET text=?, metadata=? WHERE chunk_id=?",
                [(incoming[chunk_id].text, json.dumps(incoming[chunk_id].metadata(), ensure_ascii=False), chunk_id) for chunk_id in unchanged_ids],
            )
            db.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)",
                [(chunk.chunk_id, chunk.text, json.dumps(chunk.metadata(), ensure_ascii=False), json.dumps(vector)) for chunk, vector in zip(pending, vectors)],
            )
            db.execute("INSERT OR REPLACE INTO store_metadata VALUES ('embedding_signature', ?)", (self._embedding_signature(),))

        self._invalidate_snapshot()

        return {
            "added_or_updated": len(pending),
            "unchanged": len(unchanged_ids),
            "deleted": len(deleted_ids),
        }

    def all_chunks(self) -> list[Chunk]:
        chunks, _, _ = self._snapshot()
        return list(chunks)

    def search(self, query: str, limit: int = 20, category: str | None = None) -> list[SearchResult]:
        limit = max(0, int(limit))
        if limit == 0:
            return []
        query_vector = self.embedding.embed_query(query)
        chunks, vectors, norms = self._snapshot()
        if not chunks:
            return []
        normalized_category = category.strip() if isinstance(category, str) else None
        normalized_category = normalized_category or None

        if np is not None and isinstance(vectors, np.ndarray):
            query_array = np.asarray(query_vector, dtype=np.float32)
            if vectors.ndim != 2 or query_array.ndim != 1 or vectors.shape[1] != query_array.shape[0]:
                raise RuntimeError("索引向量与当前 Embedding 模型不一致，请重新创建索引")
            query_norm = float(np.linalg.norm(query_array)) or 1.0
            denominators = norms * query_norm
            scores = np.divide(
                vectors @ query_array,
                denominators,
                out=np.zeros(len(chunks), dtype=np.float32),
                where=denominators != 0,
            )
            scores = np.maximum(scores, 0.0)
            indices = np.arange(len(chunks))
            if normalized_category:
                indices = np.fromiter(
                    (index for index, chunk in enumerate(chunks) if chunk.category == normalized_category),
                    dtype=np.int64,
                )
            if not len(indices):
                return []
            order = np.lexsort((indices, -scores[indices]))[:limit]
            selected = indices[order]
            return [
                SearchResult(chunks[int(index)], float(scores[int(index)]), vector_score=float(scores[int(index)]))
                for index in selected
            ]

        query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
        scored: list[tuple[float, int]] = []
        for index, (chunk, vector, vector_norm) in enumerate(zip(chunks, vectors, norms)):
            if normalized_category and chunk.category != normalized_category:
                continue
            denominator = query_norm * vector_norm or 1.0
            score = max(0.0, sum(left * right for left, right in zip(query_vector, vector)) / denominator)
            scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(chunks[index], score, vector_score=score)
            for score, index in scored[:limit]
        ]

    def _invalidate_snapshot(self) -> None:
        with self._snapshot_lock:
            self._cached_chunks = None
            self._cached_vectors = None
            self._cached_norms = None

    def _snapshot(self):
        """Load one immutable search snapshot and reuse it across all queries.

        The previous implementation parsed every embedding JSON value and
        scanned SQLite again for every expanded query. On medium-size libraries,
        two bilingual searches could both exceed the Agent tool deadline.
        """

        with self._snapshot_lock:
            if self._cached_chunks is not None:
                return self._cached_chunks, self._cached_vectors, self._cached_norms
            with self._connect() as db:
                rows = db.execute(
                    "SELECT chunk_id, text, metadata, embedding FROM chunks ORDER BY rowid"
                ).fetchall()
            chunks = tuple(Chunk(text=row["text"], **json.loads(row["metadata"])) for row in rows)
            vectors = [json.loads(row["embedding"]) for row in rows]
            if np is not None and vectors:
                try:
                    cached_vectors = np.asarray(vectors, dtype=np.float32)
                    if cached_vectors.ndim != 2:
                        raise ValueError("embedding matrix must be two-dimensional")
                except (MemoryError, TypeError, ValueError):
                    cached_vectors = tuple(tuple(float(value) for value in vector) for vector in vectors)
                    cached_norms = tuple(
                        math.sqrt(sum(value * value for value in vector)) for vector in cached_vectors
                    )
                else:
                    del vectors
                    cached_norms = np.sqrt(
                        np.einsum("ij,ij->i", cached_vectors, cached_vectors, optimize=True)
                    )
            else:
                cached_vectors = tuple(tuple(float(value) for value in vector) for vector in vectors)
                cached_norms = tuple(
                    math.sqrt(sum(value * value for value in vector)) for vector in cached_vectors
                )
            self._cached_chunks = chunks
            self._cached_vectors = cached_vectors
            self._cached_norms = cached_norms
            return chunks, cached_vectors, cached_norms
