"""Shared domain models."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

@dataclass(slots=True)
class DocumentPage:
    text: str
    page: int
    source_path: str
    title: str
    category: str = ""

@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    source_path: str
    page: int
    category: str
    title: str = ""

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text")
        return data

@dataclass(slots=True)
class SearchResult:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    bm25_score: float = 0.0

@dataclass(slots=True)
class Citation:
    file: str
    page: int
    path: str
    snippet: str = ""
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0

    @property
    def exists(self) -> bool:
        return Path(self.path).is_file()

@dataclass(slots=True)
class MediaItem:
    id: str
    type: str
    mime_type: str
    data_base64: str
    source_url: str = ""
    title: str = ""
    alt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class AgentResponse:
    answer: str
    sources: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    requires_review: bool = False
    review_reason: str | None = None
    thread_id: str = "default"
    run_id: str = ""
    route: str = "retrieval"
    plan: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    grounding: dict[str, Any] = field(default_factory=dict)
    harness: dict[str, Any] = field(default_factory=dict)
    media: list[MediaItem] = field(default_factory=list)
