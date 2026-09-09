"""Tools callable by the agent workflow."""
from backend.models import SearchResult
from backend.retrieval.hybrid import HybridRetriever

def search_knowledge_base(retriever: HybridRetriever, query: str, category: str | None = None) -> list[SearchResult]:
    """Search local indexed documents with optional metadata filtering."""
    return retriever.search(query, category)
