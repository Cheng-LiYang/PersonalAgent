"""LangGraph state schema."""
from collections.abc import Callable
from typing import Any, TypedDict
from backend.models import Citation, SearchResult

class AgentState(TypedDict, total=False):
    question: str
    thread_id: str
    domain: str
    keywords: list[str]
    search_queries: list[str]
    task_type: str
    route: str
    intent_analysis: dict[str, Any]
    plan: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_context: list[dict[str, Any]]
    results: list[SearchResult]
    retrieval_errors: list[str]
    answer: str
    sources: list[Citation]
    confidence: float
    grounding: dict[str, Any]
    critique: dict[str, Any]
    guardrail_flags: list[str]
    research_retry: bool
    requires_review: bool
    review_reason: str | None
    retry_count: int
    generation_error: str | None
    run_id: str
    trace_tool: Callable[[dict[str, Any]], None]
    progress: Callable[[str, str], None]
    harness: Any
