"""BGE and OpenAI embedding adapters."""
import logging
import os
from typing import Any
from backend.embedding.base import EmbeddingModel, HashEmbedding
LOGGER = logging.getLogger(__name__)

class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, model_name: str) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

def create_embedding_model(config: dict[str, Any]) -> EmbeddingModel:
    """Build configured model; use hashing when optional providers fail."""
    provider = config.get("provider", "hash")
    try:
        if provider == "openai":
            return OpenAIEmbedding(config["openai_model"])
        if provider == "local":
            return SentenceTransformerEmbedding(config["local_model"])
    except (ImportError, OSError, RuntimeError) as exc:
        LOGGER.warning("Embedding fallback activated: %s", exc)
    return HashEmbedding()
