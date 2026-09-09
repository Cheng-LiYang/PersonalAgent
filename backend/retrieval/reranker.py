"""Cross-encoder reranking with a deterministic fallback."""
import logging
from backend.models import SearchResult
from backend.retrieval.bm25 import tokenize
LOGGER = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self.model = None
        if model_name:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
            except (ImportError, OSError, RuntimeError) as exc:
                LOGGER.warning("Reranker fallback activated: %s", exc)

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not results:
            return []
        if self.model is not None:
            scores = self.model.predict([(query, item.chunk.text) for item in results])
            for result, score in zip(results, scores):
                result.score = float(score)
        else:
            query_terms = set(tokenize(query))
            for result in results:
                searchable = f"{result.chunk.source_file} {result.chunk.title} {result.chunk.category} {result.chunk.text}"
                overlap = len(query_terms & set(tokenize(searchable))) / max(1, len(query_terms))
                # Exact terminology is more trustworthy than a hash-vector
                # collision, especially for short names such as DiT or SAM.
                result.score = 0.65 * result.score + 0.35 * overlap
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
