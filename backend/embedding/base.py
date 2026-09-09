"""Embedding interface and deterministic local fallback."""
import hashlib
import math
import re
from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

class HashEmbedding(EmbeddingModel):
    """Dependency-free feature hashing for offline operation and tests."""
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower()):
                value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
                vector[value % self.dimensions] += 1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vectors.append([x / norm for x in vector])
        return vectors
