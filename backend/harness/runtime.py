"""Thread-safe runtime constraints for bounded Agent execution.

The harness is intentionally dependency free.  It centralizes four concerns that
otherwise tend to be scattered throughout an Agent graph:

* deterministic accounting for run budgets;
* cooperative cancellation and deadlines;
* composable input/tool/output policy decisions; and
* JSON-serializable state and structured failures for API/checkpoint storage.

Budget counters use a *reserve or fail* rule: a reservation that would exceed a
limit raises :class:`BudgetExceeded` and does not mutate the committed counter.
Consequently concurrent callers can never oversubscribe a budget.
"""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias


JsonObject: TypeAlias = dict[str, Any]
Clock: TypeAlias = Callable[[], float]


def _positive_number(value: float | int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number or None")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number or None")


def _nonnegative_integer(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer or None")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")


def _json_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return an immutable, detached JSON mapping or fail at construction time."""

    try:
        serialized = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
        detached = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain only JSON-serializable values") from exc
    if not isinstance(detached, dict):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(detached)


@dataclass(frozen=True, slots=True)
class RunBudget:
    """Hard limits for one Agent run.

    ``None`` disables an individual limit.  Exact-boundary consumption is valid;
    only a value strictly greater than its limit is rejected.
    """

    max_wall_time_seconds: float | None = 120.0
    max_node_steps: int | None = 32
    max_tool_calls: int | None = 16
    max_retries: int | None = 4
    max_context_chars: int | None = 120_000

    def __post_init__(self) -> None:
        _positive_number(self.max_wall_time_seconds, "max_wall_time_seconds")
        _nonnegative_integer(self.max_node_steps, "max_node_steps")
        _nonnegative_integer(self.max_tool_calls, "max_tool_calls")
        _nonnegative_integer(self.max_retries, "max_retries")
        _nonnegative_integer(self.max_context_chars, "max_context_chars")

    def snapshot(self) -> JsonObject:
        return {
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "max_node_steps": self.max_node_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_retries": self.max_retries,
            "max_context_chars": self.max_context_chars,
        }

    to_dict = snapshot


class HarnessViolation(RuntimeError):
    """Base class for machine-readable harness failures."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = copy.deepcopy(dict(details or {}))
        self.snapshot = copy.deepcopy(dict(snapshot or {}))

    def to_dict(self) -> JsonObject:
        return {
            "code": self.code,
            "message": self.message,
            "details": copy.deepcopy(self.details),
            "snapshot": copy.deepcopy(self.snapshot),
        }


class BudgetExceeded(HarnessViolation):
    """Raised atomically when a budget reservation would exceed its limit."""

    def __init__(
        self,
        resource: str,
        limit: float | int,
        observed: float | int,
        *,
        snapshot: Mapping[str, Any],
    ) -> None:
        self.resource = resource
        self.limit = limit
        self.observed = observed
        super().__init__(
            "budget_exceeded",
            f"{resource} budget exceeded: limit={limit}, observed={observed}",
            details={"resource": resource, "limit": limit, "observed": observed},
            snapshot=snapshot,
        )


class RunCancelled(HarnessViolation):
    """Raised at a checkpoint after explicit cancellation or a deadline."""

    def __init__(self, reason: str, *, deadline_exceeded: bool, snapshot: Mapping[str, Any]) -> None:
        self.reason = reason
        self.deadline_exceeded = deadline_exceeded
        code = "deadline_exceeded" if deadline_exceeded else "run_cancelled"
        super().__init__(
            code,
            f"Agent run stopped: {reason}",
            details={"reason": reason, "deadline_exceeded": deadline_exceeded},
            snapshot=snapshot,
        )


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REVIEW = "review"


class PolicyStage(str, Enum):
    INPUT = "input"
    TOOL = "tool"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """One serializable policy result."""

    action: PolicyAction
    reason: str = ""
    rule: str = ""
    stage: PolicyStage | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", PolicyAction(self.action))
        if self.stage is not None:
            object.__setattr__(self, "stage", PolicyStage(self.stage))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))

    @property
    def is_allowed(self) -> bool:
        return self.action is PolicyAction.ALLOW

    @property
    def requires_review(self) -> bool:
        return self.action is PolicyAction.REVIEW

    @classmethod
    def allow(cls, reason: str = "allowed", **kwargs: Any) -> "PolicyDecision":
        return cls(PolicyAction.ALLOW, reason=reason, **kwargs)

    @classmethod
    def deny(cls, reason: str, **kwargs: Any) -> "PolicyDecision":
        return cls(PolicyAction.DENY, reason=reason, **kwargs)

    @classmethod
    def review(cls, reason: str, **kwargs: Any) -> "PolicyDecision":
        return cls(PolicyAction.REVIEW, reason=reason, **kwargs)

    def snapshot(self) -> JsonObject:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "rule": self.rule,
            "stage": self.stage.value if self.stage is not None else None,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    to_dict = snapshot


class PolicyRule(Protocol):
    def __call__(
        self,
        stage: PolicyStage,
        subject: Any,
        context: Mapping[str, Any],
    ) -> PolicyDecision | None: ...


class PolicyEngine:
    """Compose policy rules with deterministic ``deny > review > allow`` priority.

    Rules are evaluated in registration order.  A deny short-circuits.  Otherwise
    the first review wins over all allow decisions.  ``None`` means that a rule is
    not applicable.  A rule error is converted to a fail-closed deny decision.
    """

    def __init__(self, rules: Iterable[PolicyRule] = ()) -> None:
        normalized = tuple(rules)
        if any(not callable(rule) for rule in normalized):
            raise TypeError("every policy rule must be callable")
        self._rules = normalized

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    def combined(self, *others: "PolicyEngine | PolicyRule") -> "PolicyEngine":
        rules: list[PolicyRule] = list(self._rules)
        for other in others:
            if isinstance(other, PolicyEngine):
                rules.extend(other._rules)
            elif callable(other):
                rules.append(other)
            else:
                raise TypeError("policies can only be combined with an engine or callable rule")
        return PolicyEngine(rules)

    def __or__(self, other: "PolicyEngine | PolicyRule") -> "PolicyEngine":
        return self.combined(other)

    def evaluate(
        self,
        stage: PolicyStage | str,
        subject: Any,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        resolved_stage = PolicyStage(stage)
        detached_context = copy.deepcopy(dict(context or {}))
        pending_review: PolicyDecision | None = None
        first_allow: PolicyDecision | None = None

        for index, rule in enumerate(self._rules):
            name = getattr(rule, "__name__", rule.__class__.__name__) or f"rule_{index}"
            try:
                decision = rule(resolved_stage, subject, detached_context)
            except Exception as exc:  # policy failures must fail closed
                return PolicyDecision.deny(
                    f"policy rule failed closed ({type(exc).__name__})",
                    rule=name,
                    stage=resolved_stage,
                    metadata={"error_type": type(exc).__name__},
                )
            if decision is None:
                continue
            if not isinstance(decision, PolicyDecision):
                return PolicyDecision.deny(
                    "policy rule returned an invalid decision",
                    rule=name,
                    stage=resolved_stage,
                )
            decision = replace(
                decision,
                stage=resolved_stage,
                rule=decision.rule or name,
            )
            if decision.action is PolicyAction.DENY:
                return decision
            if decision.action is PolicyAction.REVIEW and pending_review is None:
                pending_review = decision
            if decision.action is PolicyAction.ALLOW and first_allow is None:
                first_allow = decision

        if pending_review is not None:
            return pending_review
        if first_allow is not None:
            return first_allow
        return PolicyDecision.allow("no policy rule blocked the operation", stage=resolved_stage)

    def evaluate_input(self, subject: Any, context: Mapping[str, Any] | None = None) -> PolicyDecision:
        return self.evaluate(PolicyStage.INPUT, subject, context)

    def evaluate_tool(self, subject: Any, context: Mapping[str, Any] | None = None) -> PolicyDecision:
        return self.evaluate(PolicyStage.TOOL, subject, context)

    def evaluate_output(self, subject: Any, context: Mapping[str, Any] | None = None) -> PolicyDecision:
        return self.evaluate(PolicyStage.OUTPUT, subject, context)


class PolicyDenied(HarnessViolation):
    def __init__(self, decision: PolicyDecision, *, snapshot: Mapping[str, Any]) -> None:
        self.decision = decision
        super().__init__(
            "policy_denied",
            decision.reason or "operation denied by policy",
            details={"decision": decision.snapshot()},
            snapshot=snapshot,
        )


class PolicyReviewRequired(HarnessViolation):
    def __init__(self, decision: PolicyDecision, *, snapshot: Mapping[str, Any]) -> None:
        self.decision = decision
        super().__init__(
            "policy_review_required",
            decision.reason or "operation requires human review",
            details={"decision": decision.snapshot()},
            snapshot=snapshot,
        )


class CancellationToken:
    """A cooperative, thread-safe cancellation token with an optional deadline.

    ``deadline`` is an absolute value in the supplied clock's domain.  Production
    code normally uses :func:`time.monotonic`; tests may inject a deterministic
    clock.  The first cancellation reason wins.
    """

    def __init__(self, deadline: float | None = None, *, clock: Clock = time.monotonic) -> None:
        if deadline is not None:
            if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
                raise TypeError("deadline must be a finite clock value or None")
            if not math.isfinite(float(deadline)):
                raise ValueError("deadline must be a finite clock value or None")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._deadline = float(deadline) if deadline is not None else None
        self._clock = clock
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._deadline_exceeded = False

    @classmethod
    def from_timeout(cls, timeout_seconds: float, *, clock: Clock = time.monotonic) -> "CancellationToken":
        _positive_number(timeout_seconds, "timeout_seconds")
        return cls(clock() + float(timeout_seconds), clock=clock)

    @property
    def deadline(self) -> float | None:
        return self._deadline

    def _expire_if_needed(self) -> None:
        if self._deadline is None or self._event.is_set():
            return
        if self._clock() < self._deadline:
            return
        with self._lock:
            if not self._event.is_set() and self._clock() >= self._deadline:
                self._reason = "deadline exceeded"
                self._deadline_exceeded = True
                self._event.set()

    @property
    def cancelled(self) -> bool:
        self._expire_if_needed()
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        self._expire_if_needed()
        with self._lock:
            return self._reason

    @property
    def deadline_exceeded(self) -> bool:
        self._expire_if_needed()
        with self._lock:
            return self._deadline_exceeded

    def cancel(self, reason: str = "cancelled") -> bool:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancellation reason must not be empty")
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason.strip()
            self._deadline_exceeded = False
            self._event.set()
            return True

    def remaining_seconds(self) -> float | None:
        self._expire_if_needed()
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - self._clock())

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation, respecting both ``timeout`` and the deadline."""

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        self._expire_if_needed()
        if self._event.is_set():
            return True
        remaining = self.remaining_seconds()
        effective = timeout if remaining is None else remaining if timeout is None else min(timeout, remaining)
        self._event.wait(effective)
        self._expire_if_needed()
        return self._event.is_set()

    def raise_if_cancelled(self, *, snapshot: Mapping[str, Any] | None = None) -> None:
        if not self.cancelled:
            return
        raise RunCancelled(
            self.reason or "cancelled",
            deadline_exceeded=self.deadline_exceeded,
            snapshot=snapshot or {"cancellation": self.snapshot()},
        )

    def snapshot(self) -> JsonObject:
        self._expire_if_needed()
        with self._lock:
            cancelled = self._event.is_set()
            reason = self._reason
            deadline_exceeded = self._deadline_exceeded
        remaining = None if self._deadline is None else max(0.0, self._deadline - self._clock())
        return {
            "cancelled": cancelled,
            "reason": reason,
            "deadline": self._deadline,
            "deadline_exceeded": deadline_exceeded,
            "remaining_seconds": remaining,
        }


_COUNTER_LIMITS = {
    "node_steps": "max_node_steps",
    "tool_calls": "max_tool_calls",
    "retries": "max_retries",
    "context_chars": "max_context_chars",
}


class HarnessContext:
    """Run-scoped budget, cancellation and policy coordinator."""

    def __init__(
        self,
        budget: RunBudget | None = None,
        *,
        cancellation: CancellationToken | None = None,
        policy: PolicyEngine | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.budget = budget or RunBudget()
        self.cancellation = cancellation or CancellationToken(clock=clock)
        self.policy = policy or PolicyEngine()
        self._clock = clock
        self._started_at = float(clock())
        self._lock = threading.RLock()
        self._usage = {name: 0 for name in _COUNTER_LIMITS}
        self._last_policy_decision: PolicyDecision | None = None

    def _elapsed_locked(self) -> float:
        return max(0.0, float(self._clock()) - self._started_at)

    def _snapshot_locked(self) -> JsonObject:
        elapsed = self._elapsed_locked()
        budget = self.budget.snapshot()
        remaining: JsonObject = {}
        for counter, limit_name in _COUNTER_LIMITS.items():
            limit = budget[limit_name]
            remaining[counter] = None if limit is None else max(0, limit - self._usage[counter])
        wall_limit = self.budget.max_wall_time_seconds
        remaining["wall_time_seconds"] = (
            None if wall_limit is None else max(0.0, wall_limit - elapsed)
        )
        cancellation = self.cancellation.snapshot()
        return {
            "status": "cancelled" if cancellation["cancelled"] else "active",
            "started_at_clock": self._started_at,
            "elapsed_seconds": elapsed,
            "budget": budget,
            "usage": copy.deepcopy(self._usage),
            "remaining": remaining,
            "cancellation": cancellation,
            "last_policy_decision": (
                self._last_policy_decision.snapshot() if self._last_policy_decision else None
            ),
        }

    def snapshot(self) -> JsonObject:
        with self._lock:
            return self._snapshot_locked()

    def checkpoint(self) -> JsonObject:
        """Fail fast on cancellation, deadline, or wall-time exhaustion."""

        with self._lock:
            snapshot = self._snapshot_locked()
            if snapshot["cancellation"]["cancelled"]:
                raise RunCancelled(
                    snapshot["cancellation"]["reason"] or "cancelled",
                    deadline_exceeded=bool(snapshot["cancellation"]["deadline_exceeded"]),
                    snapshot=snapshot,
                )
            wall_limit = self.budget.max_wall_time_seconds
            elapsed = snapshot["elapsed_seconds"]
            if wall_limit is not None and elapsed > wall_limit:
                raise BudgetExceeded(
                    "wall_time_seconds",
                    wall_limit,
                    elapsed,
                    snapshot=snapshot,
                )
            return snapshot

    @staticmethod
    def _validate_amount(amount: int) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise TypeError("accounting amount must be a non-negative integer")
        if amount < 0:
            raise ValueError("accounting amount must be a non-negative integer")

    def _record(self, resource: str, amount: int) -> int:
        self._validate_amount(amount)
        with self._lock:
            self.checkpoint()
            current = self._usage[resource]
            observed = current + amount
            limit = getattr(self.budget, _COUNTER_LIMITS[resource])
            if limit is not None and observed > limit:
                raise BudgetExceeded(resource, limit, observed, snapshot=self._snapshot_locked())
            self._usage[resource] = observed
            return observed

    def record_node_step(self, amount: int = 1) -> int:
        return self._record("node_steps", amount)

    def record_tool_call(self, amount: int = 1) -> int:
        return self._record("tool_calls", amount)

    def record_retry(self, amount: int = 1) -> int:
        return self._record("retries", amount)

    def record_context_chars(self, amount: int) -> int:
        return self._record("context_chars", amount)

    def evaluate_policy(
        self,
        stage: PolicyStage | str,
        subject: Any,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        self.checkpoint()
        policy_context = self.snapshot()
        if context:
            policy_context["request"] = copy.deepcopy(dict(context))
        decision = self.policy.evaluate(stage, subject, policy_context)
        with self._lock:
            self._last_policy_decision = decision
        return decision

    def guard_policy(
        self,
        stage: PolicyStage | str,
        subject: Any,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        """Return allow, otherwise raise a structured deny/review exception."""

        decision = self.evaluate_policy(stage, subject, context)
        if decision.action is PolicyAction.DENY:
            raise PolicyDenied(decision, snapshot=self.snapshot())
        if decision.action is PolicyAction.REVIEW:
            raise PolicyReviewRequired(decision, snapshot=self.snapshot())
        return decision


__all__ = [
    "BudgetExceeded",
    "CancellationToken",
    "HarnessContext",
    "HarnessViolation",
    "PolicyAction",
    "PolicyDecision",
    "PolicyDenied",
    "PolicyEngine",
    "PolicyReviewRequired",
    "PolicyRule",
    "PolicyStage",
    "RunBudget",
    "RunCancelled",
]
