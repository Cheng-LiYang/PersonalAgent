"""Reciprocal hybrid fusion of vector and BM25 results."""
import threading

from backend.models import SearchResult
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.reranker import Reranker
from backend.retrieval.vector_store import VectorStore

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, config: dict) -> None:
        self.vector_store = vector_store
        self.config = config
        self.bm25 = BM25Index(vector_store.all_chunks())
        self.reranker = Reranker(config.get("reranker_model"))
        self._state_lock = threading.RLock()
        self._reranker_lock = threading.Lock()

    def refresh(self) -> None:
        refreshed = BM25Index(self.vector_store.all_chunks())
        with self._state_lock:
            self.bm25 = refreshed

    def search(self, query: str, category: str | None = None) -> list[SearchResult]:
        count = int(self.config.get("candidate_count", 20))
        vector = self.vector_store.search(query, count, category)
        with self._state_lock:
            bm25 = self.bm25
        keyword = bm25.search(query, count, category)
        merged: dict[str, SearchResult] = {}
        vector_weight = float(self.config.get("vector_weight", 0.65))
        bm25_weight = float(self.config.get("bm25_weight", 0.35))

        def normalized(items: list[SearchResult]) -> dict[str, float]:
            if not items:
                return {}
            values = [item.score for item in items]
            low, high = min(values), max(values)
            if high == low:
                return {item.chunk.chunk_id: (1.0 if high > 0 else 0.0) for item in items}
            return {item.chunk.chunk_id: (item.score - low) / (high - low) for item in items}

        vector_normalized = normalized(vector)
        keyword_normalized = normalized(keyword)
        for rank, item in enumerate(vector):
            score = vector_weight * vector_normalized[item.chunk.chunk_id] + 0.02 / (rank + 1)
            merged[item.chunk.chunk_id] = SearchResult(item.chunk, score, item.score, 0.0)
        for rank, item in enumerate(keyword):
            contribution = bm25_weight * keyword_normalized[item.chunk.chunk_id] + 0.02 / (rank + 1)
            if item.chunk.chunk_id in merged:
                current = merged[item.chunk.chunk_id]
                current.score += contribution
                current.bm25_score = item.score
            else:
                merged[item.chunk.chunk_id] = SearchResult(item.chunk, contribution, 0.0, item.score)
        candidates = sorted(merged.values(), key=lambda item: item.score, reverse=True)[:count]
        with self._reranker_lock:
            return self.reranker.rerank(query, candidates, int(self.config.get("top_k", 5)))
