"""Unicode-aware BM25 keyword retrieval."""
import math
import re
from collections import Counter

import jieba

from backend.models import Chunk, SearchResult

def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese text with jieba."""
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9_\-]+", lowered)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese = []
    for run in chinese_runs:
        chinese.extend(jieba.cut(run, cut_all=False))
    return latin + chinese

class BM25Index:
    def __init__(self, chunks: list[Chunk] | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.build(chunks or [])

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokens = [self._searchable_tokens(chunk) for chunk in chunks]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        self.document_frequency = Counter(token for terms in self.tokens for token in set(terms))

    @staticmethod
    def _searchable_tokens(chunk: Chunk) -> list[str]:
        """Include provenance so a paper can be found by its filename or title."""
        return tokenize(f"{chunk.source_file} {chunk.title} {chunk.category} {chunk.text}")

    def search(self, query: str, limit: int = 20, category: str | None = None) -> list[SearchResult]:
        scores = []
        query_terms = tokenize(query)
        total = len(self.chunks)
        for chunk, terms, length in zip(self.chunks, self.tokens, self.lengths):
            if category and chunk.category != category:
                continue
            counts = Counter(terms)
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                score += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * (1 - self.b + self.b * length / self.average_length))
            if score > 0:
                scores.append(SearchResult(chunk, score, bm25_score=score))
        return sorted(scores, key=lambda item: item.score, reverse=True)[:limit]
