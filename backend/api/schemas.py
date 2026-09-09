"""Validated API request and response schemas."""
from typing import Any

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field(default="default", min_length=1, max_length=128)

class SourceResponse(BaseModel):
    file: str
    page: int
    path: str
    snippet: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0

class MediaResponse(BaseModel):
    id: str
    type: str = "image"
    mime_type: str
    data_base64: str
    source_url: str = ""
    title: str = ""
    alt: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    confidence: float
    requires_review: bool
    review_reason: str | None
    thread_id: str
    run_id: str = ""
    route: str = "retrieval"
    plan: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    grounding: dict[str, Any] = Field(default_factory=dict)
    harness: dict[str, Any] = Field(default_factory=dict)
    media: list[MediaResponse] = Field(default_factory=list)

class IndexRequest(BaseModel):
    path: str | None = None

class ReviewRequest(BaseModel):
    thread_id: str
    approved: bool

class RunResumeRequest(BaseModel):
    approved: bool
    feedback: str | None = Field(default=None, max_length=4000)

class MemoryEpisodeRequest(BaseModel):
    """An explicit, thread-scoped long-term memory write."""

    thread_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=8000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_days: int | None = Field(default=30, ge=1, le=3650)

class ModelSettingsRequest(BaseModel):
    provider: str = Field(default="compatible", pattern="^(compatible|openai|local|extractive)$")
    model: str = Field(default="deepseek-v4-flash", min_length=1, max_length=200)
    base_url: str = Field(default="https://api.deepseek.com", min_length=1, max_length=1000)
    api_key: str = Field(default="", max_length=1000)
