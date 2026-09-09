"""FastAPI application for ingestion, indexing, chat, and HITL."""
from __future__ import annotations
import logging
import os
import threading
import uuid
from pathlib import Path
from dataclasses import asdict
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from backend.agent import KnowledgeAgent
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    IndexRequest,
    MemoryEpisodeRequest,
    ModelSettingsRequest,
    ReviewRequest,
    RunResumeRequest,
)
from backend.config import ensure_data_dirs, load_config
from backend.harness import (
    BudgetExceeded,
    HarnessViolation,
    PolicyDenied,
    PolicyReviewRequired,
    RunCancelled,
)
from backend.indexing import IndexService
from backend.runtime import JobCancelled, PersistentJobManager, QueueFullError, create_runtime
from backend.settings import SettingsStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Personal Knowledge Agent", version="2.0.0")

_INTERNAL_JOB_PHASES = {
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "cancel_requested",
    "interrupted",
}
_JOB_MANAGERS: dict[tuple[str, str, int, int, float], PersistentJobManager] = {}
_JOB_MANAGERS_LOCK = threading.Lock()
_SERVICES: tuple[dict, IndexService, KnowledgeAgent] | None = None
_SERVICES_LOCK = threading.Lock()


def _job_manager(
    data_dir: str,
    kind: str,
    max_workers: int,
    queue_capacity: int,
    completion_ttl_seconds: float,
) -> PersistentJobManager:
    """Return one durable bounded executor per data directory and job class."""

    key = (data_dir, kind, max_workers, queue_capacity, completion_ttl_seconds)
    with _JOB_MANAGERS_LOCK:
        manager = _JOB_MANAGERS.get(key)
        if manager is None:
            manager = PersistentJobManager(
                Path(data_dir) / f"{kind}_jobs.sqlite3",
                max_workers=max_workers,
                queue_capacity=queue_capacity,
                completion_ttl_seconds=completion_ttl_seconds,
            )
            _JOB_MANAGERS[key] = manager
        return manager


def _jobs(config: dict, kind: str) -> PersistentJobManager:
    settings = config.get("jobs", {})
    if kind == "index":
        workers = int(settings.get("index_workers", 1))
        capacity = int(settings.get("index_queue_capacity", 4))
    else:
        workers = int(settings.get("chat_workers", 4))
        capacity = int(settings.get("chat_queue_capacity", 32))
    return _job_manager(
        str(Path(config["app"]["data_dir"]).resolve()),
        kind,
        workers,
        capacity,
        float(settings.get("completion_ttl_seconds", 86400)),
    )


def _job_response(record: dict, kind: str) -> dict:
    """Map the durable job model onto the desktop client's legacy contract."""

    history = []
    for item in record.get("history", []):
        if item["phase"] in _INTERNAL_JOB_PHASES:
            continue
        event = {"phase": item["phase"], "detail": item.get("detail") or ""}
        if kind == "index" and item.get("progress") is not None:
            event["percent"] = int(item["progress"])
        history.append(event)
    detail = record.get("detail") or ""
    if record["status"] == "completed":
        detail = "索引创建完成" if kind == "index" else "回答生成完成"
    elif record["status"] == "failed":
        detail = "索引创建失败" if kind == "index" else "回答生成失败"
    response = {
        "task_id": record["job_id"],
        "status": record["status"],
        "phase": record["phase"],
        "detail": detail,
        "history": history,
        "result": record.get("result"),
        "error": record.get("error"),
        "cancel_requested": record.get("cancel_requested", False),
    }
    if kind == "index":
        response["percent"] = int(record.get("progress") or 0)
    return response


def _raise_harness_http_error(error: HarnessViolation) -> None:
    """Expose machine-readable Harness failures without leaking a traceback."""

    if isinstance(error, PolicyDenied):
        status_code = 403
    elif isinstance(error, BudgetExceeded):
        status_code = 429
    elif isinstance(error, RunCancelled):
        status_code = 408 if error.deadline_exceeded else 409
    elif isinstance(error, PolicyReviewRequired):
        status_code = 409
    else:
        status_code = 422
    raise HTTPException(status_code=status_code, detail=error.to_dict()) from error

def services() -> tuple[dict, IndexService, KnowledgeAgent]:
    global _SERVICES
    with _SERVICES_LOCK:
        if _SERVICES is None:
            config = load_config()
            ensure_data_dirs(config)
            log_path = Path(config["app"]["data_dir"]) / "app.log"
            if not any(
                isinstance(handler, logging.FileHandler)
                and Path(handler.baseFilename) == log_path.resolve()
                for handler in LOGGER.handlers
            ):
                file_handler = logging.FileHandler(log_path, encoding="utf-8")
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
                )
                logging.getLogger().addHandler(file_handler)
            _SERVICES = create_runtime(config)
        return _SERVICES

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/document/upload")
def upload_document(file: UploadFile = File(...), category: str = "未分类", knowledge_base: str | None = None) -> dict[str, str]:
    config, _, _ = services()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".md", ".txt", ".docx"}:
        raise HTTPException(415, "仅支持 PDF、Markdown、TXT 和 DOCX")
    safe_category = Path(category.strip()).name
    if safe_category in {"", ".", ".."}:
        raise HTTPException(400, "分类名称无效")
    safe_name = Path(file.filename or f"document{suffix}").name
    configured_root = Path(config["app"]["knowledge_base"]).expanduser().resolve()
    saved_root_value = SettingsStore(config["app"]["data_dir"]).load().get("knowledge_base")
    allowed_roots = {configured_root}
    if saved_root_value:
        allowed_roots.add(Path(saved_root_value).expanduser().resolve())
    root = Path(knowledge_base).expanduser().resolve() if knowledge_base else configured_root
    if root not in allowed_roots:
        raise HTTPException(403, "请先将该目录设为知识库并完成索引")
    if not root.is_dir():
        raise HTTPException(400, "知识库目录不存在")
    target_dir = (root / safe_category).resolve()
    if not target_dir.is_relative_to(root):
        raise HTTPException(400, "分类目录越出知识库")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = (target_dir / safe_name).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(400, "文件路径越出知识库")
    if target.exists():
        raise HTTPException(409, "同名文档已存在")
    max_upload_mb = float(config.get("api", {}).get("max_upload_mb", 50))
    if max_upload_mb <= 0:
        raise HTTPException(500, "max_upload_mb 配置必须大于 0")
    byte_limit = int(max_upload_mb * 1024 * 1024)
    temporary = target_dir / f".{uuid.uuid4().hex}.upload"
    written = 0
    try:
        with temporary.open("xb") as output:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > byte_limit:
                    raise HTTPException(413, f"文件超过 {max_upload_mb:g} MB 限制")
                output.write(chunk)
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise HTTPException(409, "同名文档已存在") from exc
        except OSError:
            # Some filesystems do not support hard links. Exclusive creation
            # still prevents overwriting, while the scanner ignores .upload.
            with temporary.open("rb") as source, target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except FileExistsError as exc:
        raise HTTPException(409, "同名文档已存在") from exc
    except OSError as exc:
        LOGGER.exception("Upload failed")
        target.unlink(missing_ok=True)
        raise HTTPException(500, str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {"file": safe_name, "path": str(target.resolve()), "knowledge_base": str(root)}

@app.post("/index/build")
def build_index(request: IndexRequest) -> dict:
    config, indexer, agent = services()
    root = Path(request.path or config["app"]["knowledge_base"]).resolve()
    if not root.is_dir():
        raise HTTPException(400, "知识库目录不存在")
    try:
        with _jobs(config, "index").reserve():
            result = indexer.build(root)
        agent.retriever.refresh()
        SettingsStore(config["app"]["data_dir"]).save_knowledge_base(str(root))
        return result
    except QueueFullError as exc:
        raise HTTPException(429, str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.exception("Indexing failed")
        raise HTTPException(500, str(exc)) from exc

@app.post("/index/build/start")
def start_index_build(request: IndexRequest) -> dict[str, str]:
    """Start indexing in the background and return a progress task id."""
    config, indexer, agent = services()
    root = Path(request.path or config["app"]["knowledge_base"]).resolve()
    if not root.is_dir():
        raise HTTPException(400, "知识库目录不存在")
    def run_build(update, cancelled) -> dict:
        try:
            def report(phase: str, percent: int, detail: str) -> None:
                if not update(phase, percent, detail):
                    raise JobCancelled("索引任务已取消")

            result = indexer.build(root, report)
            if cancelled():
                raise JobCancelled("索引任务已取消")
            agent.retriever.refresh()
            SettingsStore(config["app"]["data_dir"]).save_knowledge_base(str(root))
            return result
        except Exception:
            LOGGER.exception("Background indexing failed")
            raise

    try:
        task_id = _jobs(config, "index").submit("index", run_build, metadata={"root": str(root)})
    except QueueFullError as exc:
        raise HTTPException(429, str(exc)) from exc
    return {"task_id": task_id}

@app.get("/index/status/{task_id}")
def index_build_status(task_id: str) -> dict:
    config, _, _ = services()
    job = _jobs(config, "index").get(task_id)
    if not job:
        raise HTTPException(404, "索引任务不存在")
    return _job_response(job, "index")

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    config, _, agent = services()
    try:
        with _jobs(config, "chat").reserve():
            result = agent.ask(request.question, request.thread_id)
        return ChatResponse(
            answer=result.answer,
            sources=[{"file": s.file, "page": s.page, "path": s.path, "snippet": s.snippet, "vector_score": s.vector_score, "bm25_score": s.bm25_score, "hybrid_score": s.hybrid_score} for s in result.sources],
            confidence=result.confidence,
            requires_review=result.requires_review,
            review_reason=result.review_reason,
            thread_id=result.thread_id,
            run_id=result.run_id,
            route=result.route,
            plan=result.plan,
            tool_calls=result.tool_calls,
            grounding=result.grounding,
            harness=result.harness,
            media=[asdict(item) for item in result.media],
        )
    except QueueFullError as exc:
        raise HTTPException(429, str(exc)) from exc
    except HarnessViolation as exc:
        _raise_harness_http_error(exc)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.exception("Chat failed")
        raise HTTPException(500, str(exc)) from exc

def _chat_response(result) -> dict:
    return ChatResponse(
        answer=result.answer,
        sources=[{
            "file": source.file,
            "page": source.page,
            "path": source.path,
            "snippet": source.snippet,
            "vector_score": source.vector_score,
            "bm25_score": source.bm25_score,
            "hybrid_score": source.hybrid_score,
        } for source in result.sources],
        confidence=result.confidence,
        requires_review=result.requires_review,
        review_reason=result.review_reason,
        thread_id=result.thread_id,
        run_id=result.run_id,
        route=result.route,
        plan=result.plan,
        tool_calls=result.tool_calls,
        grounding=result.grounding,
        harness=result.harness,
        media=[asdict(item) for item in result.media],
    ).model_dump()

@app.post("/chat/start")
def start_chat(request: ChatRequest) -> dict[str, str]:
    """Start a chat turn and expose its real workflow phase for the UI."""
    config, _, agent = services()

    def run_chat(update, cancelled) -> dict:
        try:
            def report(phase: str, detail: str) -> None:
                if not update(phase, detail):
                    raise JobCancelled("对话任务已取消")

            result = agent.ask(request.question, request.thread_id, progress=report)
            if cancelled():
                raise JobCancelled("对话任务已取消")
            return _chat_response(result)
        except Exception:
            LOGGER.exception("Background chat failed")
            raise

    try:
        task_id = _jobs(config, "chat").submit(
            "chat",
            run_chat,
            metadata={"thread_id": request.thread_id},
        )
    except QueueFullError as exc:
        raise HTTPException(429, str(exc)) from exc
    return {"task_id": task_id}

@app.get("/chat/status/{task_id}")
def chat_status(task_id: str) -> dict:
    config, _, _ = services()
    job = _jobs(config, "chat").get(task_id)
    if not job:
        raise HTTPException(404, "对话任务不存在")
    return _job_response(job, "chat")

@app.post("/jobs/{task_id}/cancel")
def cancel_job(task_id: str) -> dict[str, bool]:
    """Request cooperative cancellation for either background job class."""

    config, _, _ = services()
    for kind in ("chat", "index"):
        manager = _jobs(config, kind)
        if manager.get(task_id, include_history=False) is not None:
            return {"cancelled": manager.cancel(task_id)}
    raise HTTPException(404, "后台任务不存在")

@app.get("/jobs")
def list_jobs(
    status: str | None = None,
    kind: str | None = Query(default=None, pattern="^(chat|index)$"),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """List durable jobs for operational inspection."""

    config, _, _ = services()
    kinds = (kind,) if kind else ("chat", "index")
    try:
        items = [
            item
            for selected in kinds
            for item in _jobs(config, selected).list(status=status, kind=selected, limit=limit)
        ]
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    items.sort(key=lambda item: (item["created_at"], item["job_id"]), reverse=True)
    return {"items": items[:limit]}

@app.post("/review")
def review(request: ReviewRequest) -> dict[str, bool]:
    _, _, agent = services()
    return {"resolved": agent.resolve_review(request.thread_id, request.approved)}

@app.post("/memory/episodes", status_code=201)
def remember_episode(request: MemoryEpisodeRequest) -> dict:
    """Persist an explicit episodic memory inside one conversation scope."""

    _, _, agent = services()
    episode_id = agent.memory.remember_episode(
        request.thread_id,
        request.content,
        metadata=request.metadata,
        importance=request.importance,
        ttl_days=request.ttl_days,
    )
    return {"id": episode_id, "thread_id": request.thread_id, "stored": True}

@app.get("/memory/episodes/search")
def search_episodes(
    thread_id: str = Query(min_length=1, max_length=128),
    query: str = Query(min_length=1, max_length=1000),
    limit: int = Query(default=5, ge=1, le=20),
) -> dict:
    """Search only non-expired memories belonging to the requested thread."""

    _, _, agent = services()
    return {"items": agent.memory.search_episodes(thread_id, query, limit)}

@app.get("/runs/{run_id}")
def get_agent_run(run_id: str) -> dict:
    """Return a durable run with node spans, tool events, and review state."""

    _, _, agent = services()
    record = agent.traces.get_run(run_id)
    if record is None:
        raise HTTPException(404, "Agent 运行记录不存在")
    return record

@app.post("/runs/{run_id}/resume")
def resume_agent_run(run_id: str, request: RunResumeRequest) -> dict:
    """Resume a persisted HITL checkpoint using a run-scoped decision."""

    _, _, agent = services()
    try:
        return agent.resume_run(run_id, request.approved, request.feedback)
    except KeyError as exc:
        raise HTTPException(404, "Agent 运行记录或审核点不存在") from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

@app.get("/observability/runs")
def list_agent_runs(status: str | None = None, thread_id: str | None = None, limit: int = 50) -> dict:
    """List recent Agent runs for debugging and bad-case analysis."""

    _, _, agent = services()
    try:
        return {"items": agent.traces.list_runs(status=status, thread_id=thread_id, limit=limit)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.get("/observability/metrics")
def agent_metrics(since: str | None = None) -> dict:
    """Expose lightweight run/tool success and latency aggregates."""

    _, _, agent = services()
    return agent.traces.metrics(since=since)

@app.post("/settings/model")
def configure_model(request: ModelSettingsRequest) -> dict[str, str]:
    """Apply a model and persist its secret with the platform settings store."""
    _, _, agent = services()
    settings_store = SettingsStore(services()[0]["app"]["data_dir"])
    saved = settings_store.load_model() or {}
    api_key = request.api_key or saved.get("api_key", "")
    if request.provider != "extractive" and not api_key and request.provider != "local":
        raise HTTPException(400, "云端模型必须提供 API Key")
    model_config = {**request.model_dump(), "api_key": api_key}
    try:
        agent.configure_model(model_config)
    except Exception as exc:
        LOGGER.warning("Model connection failed for %s: %s", request.model, exc)
        raise HTTPException(400, f"模型连接失败：{exc}") from exc
    settings_store.save_model(model_config, api_key)
    return {"status": "configured", "provider": request.provider, "model": request.model}

@app.get("/settings")
def get_saved_settings() -> dict:
    config, indexer, _ = services()
    store = SettingsStore(config["app"]["data_dir"])
    saved = store.load()
    model = saved.get("model") or {"provider": "extractive", "model": "offline-extractive", "base_url": "http://127.0.0.1"}
    return {
        "model": model,
        "has_api_key": bool(saved.get("api_key_encrypted")),
        "knowledge_base": saved.get("knowledge_base", config["app"]["knowledge_base"]),
        "indexed_chunks": len(indexer.store.all_chunks()),
        "data_dir": config["app"]["data_dir"],
    }

def run() -> None:
    """Start the development API server."""
    import uvicorn
    config = load_config()["api"]
    uvicorn.run("backend.api.server:app", host=config["host"], port=int(config["port"]), reload=False)

if __name__ == "__main__":
    run()
