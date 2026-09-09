"""Run-scoped integration between runtime, replay, and chaos harnesses."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import threading
from typing import Any, Mapping

from backend.harness.replay import FaultInjector, Recorder, ReplaySession
from backend.harness.runtime import (
    HarnessContext,
    PolicyAction,
    PolicyDecision,
    PolicyDenied,
    PolicyStage,
    RunBudget,
)


def _plan_view(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove generated IDs while retaining replay-relevant tool intent."""

    return [
        {
            "tool": str(step.get("tool", "")),
            "arguments": copy.deepcopy(dict(step.get("arguments", {}))),
            "depends_on": list(step.get("depends_on", [])),
        }
        for step in plan
    ]


def _state_view(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic and JSON-safe graph-state summary."""

    critique = state.get("critique", {})
    return {
        "route": state.get("route"),
        "domain": state.get("domain"),
        "task_type": state.get("task_type"),
        "retry_count": int(state.get("retry_count", 0)),
        "plan": _plan_view(list(state.get("plan", []))),
        "tool_call_count": len(state.get("tool_calls", [])),
        "result_count": len(state.get("results", [])),
        "confidence": state.get("confidence"),
        "verdict": critique.get("verdict") if isinstance(critique, Mapping) else None,
        "requires_review": state.get("requires_review"),
    }


def _tool_call_view(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep stable tool fields and omit latency, generated call IDs, and objects."""

    return [
        {
            "name": str(call.get("name", call.get("tool", ""))),
            "arguments": copy.deepcopy(dict(call.get("arguments", {}))),
            "status": call.get("status"),
            "error": call.get("error"),
        }
        for call in calls
    ]


def _response_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a response for deterministic cassette comparison."""

    sources = []
    for source in payload.get("sources", []):
        if isinstance(source, Mapping):
            sources.append(
                {
                    "file": source.get("file"),
                    "page": source.get("page"),
                    "path": source.get("path"),
                    "snippet": source.get("snippet"),
                }
            )
    return {
        "answer": payload.get("answer"),
        "sources": sources,
        "confidence": payload.get("confidence"),
        "requires_review": payload.get("requires_review"),
        "review_reason": payload.get("review_reason"),
        "route": payload.get("route"),
        "plan": _plan_view(list(payload.get("plan", []))),
        "tool_calls": _tool_call_view(list(payload.get("tool_calls", []))),
        "grounding": copy.deepcopy(dict(payload.get("grounding", {}))),
    }


@dataclass(slots=True)
class HarnessSession:
    """Coordinate all harness concerns for exactly one Agent run.

    A session is intentionally not reusable: counters, cassette cursors and
    fault occurrences are all scoped to one request.  Recorder and replay are
    mutually exclusive because one run either establishes or verifies a golden
    execution contract.
    """

    context: HarnessContext = field(default_factory=HarnessContext)
    recorder: Recorder | None = None
    replay: ReplaySession | None = None
    faults: FaultInjector | None = None
    _started: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _status: str = field(default="created", init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.recorder is not None and self.replay is not None:
            raise ValueError("recorder and replay cannot be enabled in the same session")

    def _observe(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self.recorder is not None:
            self.recorder.record(kind, payload)
        if self.replay is not None:
            self.replay.consume(kind, payload)

    def _inject(self, kind: str, payload: Any) -> Any:
        if self.faults is None:
            return payload
        return self.faults.before(kind, payload)

    def start(self, question: str, thread_id: str, *, run_id: str | None = None) -> str:
        """Validate input policy, apply input faults, and open the event stream."""

        with self._lock:
            if self._started:
                raise RuntimeError("harness session has already started")
            if self._closed:
                raise RuntimeError("harness session is closed")
            if run_id is not None:
                if not isinstance(run_id, str) or not run_id.strip():
                    raise ValueError("run_id must be a non-empty string")
                if self.recorder is not None:
                    # No event has been flushed yet, so the cassette header can
                    # share the durable Observability run identifier.
                    self.recorder.cassette.run_id = run_id
            self.context.checkpoint()
            self.context.guard_policy(
                PolicyStage.INPUT,
                question,
                {"thread_id": thread_id},
            )
            effective = self._inject("run.input", question)
            if not isinstance(effective, str) or not effective.strip():
                raise TypeError("run.input fault must leave a non-empty string question")
            self._observe("run.started", {"question": effective, "thread_id": thread_id})
            self._started = True
            self._status = "running"
            return effective

    def before_node(self, name: str, state: Mapping[str, Any]) -> None:
        self.context.record_node_step()
        payload = {"node": name, "state": _state_view(state)}
        injected = self._inject(f"node.{name}", payload)
        if not isinstance(injected, Mapping):
            raise TypeError("node fault must leave a mapping payload")
        self._observe("node.before", dict(injected))

    def after_node(self, name: str, state: Mapping[str, Any]) -> None:
        self.context.checkpoint()
        self._observe("node.after", {"node": name, "state": _state_view(state)})

    def node_failed(self, name: str, error: BaseException) -> None:
        self._observe(
            "node.failed",
            {"node": name, "error_type": type(error).__name__, "message": str(error)},
        )

    def before_tools(
        self,
        plan: list[dict[str, Any]],
        state: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Atomically reserve calls, guard each tool, then apply chaos faults."""

        self.context.record_tool_call(len(plan))
        for step in plan:
            subject = {
                "tool": str(step.get("tool", "")),
                "arguments": copy.deepcopy(dict(step.get("arguments", {}))),
            }
            self.context.guard_policy(
                PolicyStage.TOOL,
                subject,
                {"retry_count": int(state.get("retry_count", 0))},
            )
        effective = self._inject("tools.execute", copy.deepcopy(plan))
        if not isinstance(effective, list) or any(not isinstance(item, dict) for item in effective):
            raise TypeError("tools.execute fault must leave a list of plan objects")
        self._observe("tools.before", {"plan": _plan_view(effective)})
        return effective

    def after_tools(self, calls: list[dict[str, Any]]) -> None:
        self.context.checkpoint()
        self._observe("tools.after", {"calls": _tool_call_view(calls)})

    def context_built(
        self,
        stats: Mapping[str, Any],
        *,
        evidence: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[dict[str, str]]] | None:
        evidence_chars = int(stats.get("evidence_chars", 0))
        history_chars = int(stats.get("history_chars", 0))
        self.context.record_context_chars(evidence_chars + history_chars)
        payload = {
            "evidence": evidence,
            "history": copy.deepcopy(history),
            "stats": copy.deepcopy(dict(stats)),
        }
        injected = self._inject("context.build", payload)
        if not isinstance(injected, Mapping):
            raise TypeError("context.build fault must leave a mapping payload")
        injected_stats = injected.get("stats", stats)
        if not isinstance(injected_stats, Mapping):
            raise TypeError("context.build stats must remain a mapping")
        # Record only sizes and counts; cassettes should not duplicate private
        # knowledge-base evidence or conversation content.
        self._observe("context.built", {"stats": copy.deepcopy(dict(injected_stats))})
        if evidence is None and history is None:
            return None
        effective_evidence = injected.get("evidence")
        effective_history = injected.get("history")
        if not isinstance(effective_evidence, str):
            raise TypeError("context.build evidence must remain a string")
        if not isinstance(effective_history, list) or any(
            not isinstance(item, dict) for item in effective_history
        ):
            raise TypeError("context.build history must remain a list of objects")
        return effective_evidence, copy.deepcopy(effective_history)

    def retry(self, kind: str, retry_count: int) -> None:
        self.context.record_retry()
        payload = {"kind": kind, "retry_count": retry_count}
        injected = self._inject(f"retry.{kind}", payload)
        if not isinstance(injected, Mapping):
            raise TypeError("retry fault must leave a mapping payload")
        self._observe("run.retry", dict(injected))

    def evaluate_output(self, payload: Mapping[str, Any]) -> PolicyDecision:
        """Deny unsafe output, while allowing the caller to route review to HITL."""

        effective = self._inject("run.output", copy.deepcopy(dict(payload)))
        if not isinstance(effective, Mapping):
            raise TypeError("run.output fault must leave a mapping payload")
        decision = self.context.evaluate_policy(PolicyStage.OUTPUT, effective)
        if decision.action is PolicyAction.DENY:
            raise PolicyDenied(decision, snapshot=self.context.snapshot())
        return decision

    def complete(self, payload: Mapping[str, Any]) -> None:
        self.context.checkpoint()
        self._observe("run.completed", _response_view(payload))
        self._status = "completed"
        self.close()

    def fail(self, error: BaseException) -> None:
        if self._closed:
            return
        self._status = "failed"
        self._observe(
            "run.failed",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        self.close(verify_replay=False)

    def close(self, *, verify_replay: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            if self.recorder is not None:
                self.recorder.close()
            if self.replay is not None:
                if verify_replay:
                    self.replay.close()
                else:
                    self.replay.verify_on_close = False
                    self.replay.close()
            self._closed = True

    def snapshot(self) -> dict[str, Any]:
        faults = None
        if self.faults is not None:
            faults = {
                "hit_count": self.faults.hit_count,
                "hit_counts": list(self.faults.hit_counts),
                "history": [asdict(item) for item in self.faults.history],
            }
        return {
            "status": self._status,
            "runtime": self.context.snapshot(),
            "recording": {
                "enabled": self.recorder is not None,
                "path": str(self.recorder.path) if self.recorder is not None else None,
                "run_id": self.recorder.cassette.run_id if self.recorder is not None else None,
                "event_count": len(self.recorder.cassette) if self.recorder is not None else 0,
            },
            "replay": {
                "enabled": self.replay is not None,
                "recorded_run_id": self.replay.cassette.run_id if self.replay is not None else None,
                "cursor": self.replay.cursor if self.replay is not None else 0,
                "remaining": self.replay.remaining if self.replay is not None else 0,
            },
            "faults": faults,
        }


def session_from_config(config: Mapping[str, Any]) -> HarnessSession | None:
    """Create the default runtime harness configured for each Agent request."""

    settings = dict(config.get("harness", {}))
    if settings.get("enabled", True) is False:
        return None
    budget = RunBudget(
        max_wall_time_seconds=settings.get("max_wall_time_seconds", 120.0),
        max_node_steps=settings.get("max_node_steps", 32),
        max_tool_calls=settings.get("max_tool_calls", 16),
        max_retries=settings.get("max_retries", 4),
        max_context_chars=settings.get("max_context_chars", 120_000),
    )
    return HarnessSession(HarnessContext(budget))


def coerce_session(value: HarnessSession | HarnessContext | None) -> HarnessSession | None:
    if value is None or isinstance(value, HarnessSession):
        return value
    if isinstance(value, HarnessContext):
        return HarnessSession(value)
    raise TypeError("harness must be HarnessSession, HarnessContext, or None")


def recording_session(
    path: str | Path,
    *,
    budget: RunBudget | None = None,
    faults: FaultInjector | None = None,
    overwrite: bool = False,
) -> HarnessSession:
    """Convenience factory for one recorded run."""

    return HarnessSession(
        HarnessContext(budget),
        recorder=Recorder(path, overwrite=overwrite),
        faults=faults,
    )


def replay_session(
    path: str | Path,
    *,
    budget: RunBudget | None = None,
    faults: FaultInjector | None = None,
) -> HarnessSession:
    """Convenience factory for strict workflow-contract replay."""

    return HarnessSession(
        HarnessContext(budget),
        replay=ReplaySession(path),
        faults=faults,
    )


__all__ = [
    "HarnessSession",
    "coerce_session",
    "recording_session",
    "replay_session",
    "session_from_config",
]
