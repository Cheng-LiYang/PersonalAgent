"""Typed, dependency-free tool registration and execution primitives.

The registry deliberately exposes results as data instead of raising tool errors.
This keeps an agent loop in control of retries, approval, and termination.  Timeouts
are cooperative at the process level: Python cannot safely stop a running thread,
so a timed-out handler may finish in the background and side-effecting handlers
should also implement their own cancellation or idempotency guarantees.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias


ToolHandler: TypeAlias = Callable[..., Any]
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_JSON_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}
_GLOBAL_TOOL_SLOTS = threading.BoundedSemaphore(32)


class RiskLevel(str, Enum):
    """Coarse risk attached to a tool before its arguments are executed."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_RISK_RANK = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class ToolStatus(str, Enum):
    SUCCESS = "success"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    APPROVAL_REQUIRED = "approval_required"
    DENIED = "denied"
    ERROR = "error"
    TIMEOUT = "timeout"
    CALL_LIMIT_EXCEEDED = "call_limit_exceeded"
    CAPACITY_EXCEEDED = "capacity_exceeded"


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """Decide whether risk needs approval or must be denied outright.

    ``None`` disables the corresponding threshold.  An explicit
    ``ToolSpec.requires_approval`` still requires approval even when the policy's
    approval threshold is disabled.
    """

    require_at_or_above: RiskLevel | None = RiskLevel.HIGH
    deny_at_or_above: RiskLevel | None = None

    def __post_init__(self) -> None:
        if self.require_at_or_above is not None:
            object.__setattr__(self, "require_at_or_above", RiskLevel(self.require_at_or_above))
        if self.deny_at_or_above is not None:
            object.__setattr__(self, "deny_at_or_above", RiskLevel(self.deny_at_or_above))

    def decision(self, spec: ToolSpec, approved: bool | None) -> ToolStatus | None:
        risk_rank = _RISK_RANK[spec.risk]
        if self.deny_at_or_above is not None and risk_rank >= _RISK_RANK[self.deny_at_or_above]:
            return ToolStatus.DENIED
        needs_approval = spec.requires_approval
        if self.require_at_or_above is not None:
            needs_approval = needs_approval or risk_rank >= _RISK_RANK[self.require_at_or_above]
        if not needs_approval:
            return None
        if approved is True:
            return None
        return ToolStatus.DENIED if approved is False else ToolStatus.APPROVAL_REQUIRED


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Serializable description of one callable tool."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    risk: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False
    timeout_seconds: float | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        if not _TOOL_NAME.fullmatch(self.name):
            raise ValueError("tool name must contain 1-64 letters, digits, underscores, or hyphens")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be a JSON schema object")
        parameters = copy.deepcopy(dict(self.parameters))
        parameters.setdefault("type", "object")
        root_types = _schema_types(parameters.get("type"), "$.type")
        if "object" not in root_types:
            raise ValueError("tool parameter schema must accept an object at its root")
        _validate_schema_definition(parameters)
        if self.strict:
            properties = set(parameters.get("properties", {}))
            required = set(parameters.get("required", []))
            if parameters.get("additionalProperties") is not False or properties != required:
                raise ValueError(
                    "strict tool schemas require additionalProperties=false and every property to be required"
                )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "risk", RiskLevel(self.risk))
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("tool timeout_seconds must be positive")

    def to_openai_tool(self) -> dict[str, Any]:
        """Return the OpenAI-compatible function-tools representation."""

        function = {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(dict(self.parameters)),
        }
        if self.strict:
            function["strict"] = True
        return {
            "type": "function",
            "function": function,
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any] | str = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ValueError("tool call_id must not be empty")
        if not self.name:
            raise ValueError("tool name must not be empty")
        if isinstance(self.arguments, Mapping):
            object.__setattr__(self, "arguments", copy.deepcopy(dict(self.arguments)))


@dataclass(frozen=True, slots=True)
class ToolExecution:
    call: ToolCall
    status: ToolStatus
    output: Any = None
    error: str | None = None
    risk: RiskLevel | None = None
    approved: bool | None = None
    started_at: float | None = None
    finished_at: float | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler


@dataclass(frozen=True, slots=True)
class _PreparedCall:
    call: ToolCall
    registered: _RegisteredTool
    arguments: dict[str, Any]
    approved: bool | None
    timeout_seconds: float | None


def _schema_types(declaration: Any, path: str) -> tuple[str, ...]:
    if isinstance(declaration, str):
        values = (declaration,)
    elif isinstance(declaration, Sequence) and not isinstance(declaration, (str, bytes)):
        values = tuple(declaration)
    else:
        raise ValueError(f"{path} must be a JSON schema type or list of types")
    if not values or any(not isinstance(value, str) or value not in _JSON_TYPES for value in values):
        raise ValueError(f"{path} contains an unsupported JSON schema type")
    return values


def _validate_schema_definition(schema: Mapping[str, Any], path: str = "$") -> None:
    if "type" in schema:
        _schema_types(schema["type"], f"{path}.type")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}.properties must be an object")
        for key, nested in properties.items():
            if not isinstance(key, str) or not isinstance(nested, Mapping):
                raise ValueError(f"{path}.properties must map string names to schemas")
            _validate_schema_definition(nested, f"{path}.properties.{key}")
    required = schema.get("required")
    if required is not None and (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes))
        or any(not isinstance(item, str) for item in required)
    ):
        raise ValueError(f"{path}.required must be a list of strings")
    additional = schema.get("additionalProperties", True)
    if not isinstance(additional, (bool, Mapping)):
        raise ValueError(f"{path}.additionalProperties must be a boolean or schema")
    if isinstance(additional, Mapping):
        _validate_schema_definition(additional, f"{path}.additionalProperties")
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ValueError(f"{path}.items must be a schema object")
        _validate_schema_definition(items, f"{path}.items")
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, Sequence) or isinstance(enum, (str, bytes))):
        raise ValueError(f"{path}.enum must be a list")
    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ValueError(f"{path}.pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"{path}.pattern is invalid: {exc}") from exc


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, Mapping)
    return False


def _validate_value(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    if "type" in schema:
        allowed = _schema_types(schema["type"], f"{path}.type")
        if not any(_matches_type(value, type_name) for type_name in allowed):
            errors.append(f"{path} must be {' or '.join(allowed)}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {list(schema['enum'])!r}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path} must equal {schema['const']!r}")

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is required")
        additional = schema.get("additionalProperties", True)
        for key, nested_value in value.items():
            key_path = f"{path}.{key}"
            if key in properties:
                _validate_value(nested_value, properties[key], key_path, errors)
            elif additional is False:
                errors.append(f"{key_path} is not allowed")
            elif isinstance(additional, Mapping):
                _validate_value(nested_value, additional, key_path, errors)

    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if minimum is not None and len(value) < minimum:
            errors.append(f"{path} must contain at least {minimum} items")
        if maximum is not None and len(value) > maximum:
            errors.append(f"{path} must contain at most {maximum} items")
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _validate_value(item, items, f"{path}[{index}]", errors)

    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if minimum is not None and len(value) < minimum:
            errors.append(f"{path} must contain at least {minimum} characters")
        if maximum is not None and len(value) > maximum:
            errors.append(f"{path} must contain at most {maximum} characters")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            errors.append(f"{path} must match {pattern!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} must be <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path} must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path} must be < {schema['exclusiveMaximum']}")


def validate_json_schema(value: Any, schema: Mapping[str, Any]) -> tuple[str, ...]:
    """Return basic JSON-schema validation errors without third-party packages."""

    errors: list[str] = []
    _validate_value(value, schema, "$", errors)
    return tuple(errors)


class ToolRegistry:
    """Register typed tools and execute calls under hard per-batch bounds."""

    def __init__(
        self,
        *,
        max_calls: int = 8,
        max_parallel: int = 4,
        default_timeout_seconds: float | None = 30.0,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        if max_calls < 0:
            raise ValueError("max_calls must be non-negative")
        if max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        if default_timeout_seconds is not None and default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        self.max_calls = max_calls
        self.max_parallel = max_parallel
        self.default_timeout_seconds = default_timeout_seconds
        self.approval_policy = approval_policy or ApprovalPolicy()
        self._tools: dict[str, _RegisteredTool] = {}
        self._lock = threading.RLock()

    def register(self, spec: ToolSpec, handler: ToolHandler) -> ToolSpec:
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        with self._lock:
            if spec.name in self._tools:
                raise ValueError(f"tool {spec.name!r} is already registered")
            self._tools[spec.name] = _RegisteredTool(spec, handler)
        return spec

    def get(self, name: str) -> ToolSpec | None:
        with self._lock:
            registered = self._tools.get(name)
            return registered.spec if registered else None

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        with self._lock:
            return tuple(item.spec for item in self._tools.values())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [spec.to_openai_tool() for spec in self.specs]

    def execute(
        self,
        call: ToolCall,
        *,
        approved: bool | None = None,
        timeout_seconds: float | None = None,
    ) -> ToolExecution:
        approvals = {call.call_id: approved} if approved is not None else None
        return self.execute_many(
            [call], approvals=approvals, max_calls=1, max_parallel=1, timeout_seconds=timeout_seconds
        )[0]

    def execute_many(
        self,
        calls: Iterable[ToolCall],
        *,
        approvals: Mapping[str, bool] | None = None,
        max_calls: int | None = None,
        max_parallel: int | None = None,
        timeout_seconds: float | None = None,
    ) -> list[ToolExecution]:
        """Execute calls in parallel while preserving input order.

        Overrides may only tighten the registry's hard call and worker limits.
        Timeout includes time waiting in the bounded worker queue.
        """

        call_list = list(calls)
        requested_calls = self.max_calls if max_calls is None else max_calls
        requested_parallel = self.max_parallel if max_parallel is None else max_parallel
        if requested_calls < 0:
            raise ValueError("max_calls must be non-negative")
        if requested_parallel < 1:
            raise ValueError("max_parallel must be positive")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        call_limit = min(self.max_calls, requested_calls)
        parallel_limit = min(self.max_parallel, requested_parallel)
        results: list[ToolExecution | None] = [None] * len(call_list)
        runnable: list[tuple[int, _PreparedCall]] = []

        for index, call in enumerate(call_list):
            if index >= call_limit:
                spec = self.get(call.name)
                results[index] = self._instant_result(
                    call,
                    ToolStatus.CALL_LIMIT_EXCEEDED,
                    "tool call budget exceeded",
                    risk=spec.risk if spec else None,
                )
                continue
            approval = approvals.get(call.call_id) if approvals is not None else None
            prepared = self._prepare(call, approval, timeout_seconds)
            if isinstance(prepared, ToolExecution):
                results[index] = prepared
            else:
                runnable.append((index, prepared))

        if runnable:
            executor = ThreadPoolExecutor(
                max_workers=min(parallel_limit, len(runnable)),
                thread_name_prefix="agent-tool",
            )
            futures: list[tuple[int, _PreparedCall, Future[ToolExecution], float, float]] = []
            try:
                for index, prepared in runnable:
                    submitted_wall = time.time()
                    submitted_monotonic = time.monotonic()
                    future = executor.submit(self._invoke, prepared)
                    futures.append((index, prepared, future, submitted_wall, submitted_monotonic))
                for index, prepared, future, submitted_wall, submitted_monotonic in futures:
                    timeout = prepared.timeout_seconds
                    remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - submitted_monotonic))
                    try:
                        results[index] = future.result(timeout=remaining)
                    except FutureTimeoutError:
                        future.cancel()
                        finished = time.time()
                        results[index] = ToolExecution(
                            call=prepared.call,
                            status=ToolStatus.TIMEOUT,
                            error=f"tool exceeded timeout of {timeout:.3f}s",
                            risk=prepared.registered.spec.risk,
                            approved=prepared.approved,
                            started_at=submitted_wall,
                            finished_at=finished,
                            duration_ms=max(0.0, (finished - submitted_wall) * 1000),
                        )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

        return [result for result in results if result is not None]

    def _prepare(
        self,
        call: ToolCall,
        approved: bool | None,
        timeout_override: float | None,
    ) -> _PreparedCall | ToolExecution:
        with self._lock:
            registered = self._tools.get(call.name)
        if registered is None:
            return self._instant_result(call, ToolStatus.UNKNOWN_TOOL, f"unknown tool: {call.name}")
        arguments, parse_error = self._parse_arguments(call.arguments)
        if parse_error:
            return self._instant_result(
                call, ToolStatus.INVALID_ARGUMENTS, parse_error, risk=registered.spec.risk, approved=approved
            )
        errors = validate_json_schema(arguments, registered.spec.parameters)
        if errors:
            return self._instant_result(
                call,
                ToolStatus.INVALID_ARGUMENTS,
                "; ".join(errors),
                risk=registered.spec.risk,
                approved=approved,
            )
        decision = self.approval_policy.decision(registered.spec, approved)
        if decision is not None:
            message = "tool execution was denied" if decision is ToolStatus.DENIED else "tool requires approval"
            return self._instant_result(
                call, decision, message, risk=registered.spec.risk, approved=approved
            )
        timeout_candidates = [
            value
            for value in (self.default_timeout_seconds, registered.spec.timeout_seconds, timeout_override)
            if value is not None
        ]
        timeout = min(timeout_candidates) if timeout_candidates else None
        return _PreparedCall(call, registered, arguments, approved, timeout)

    @staticmethod
    def _parse_arguments(arguments: Any) -> tuple[dict[str, Any], str | None]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                return {}, f"arguments are not valid JSON: {exc}"
        if not isinstance(arguments, Mapping):
            return {}, "tool arguments must be a JSON object"
        return copy.deepcopy(dict(arguments)), None

    @staticmethod
    def _invoke(prepared: _PreparedCall) -> ToolExecution:
        started_wall = time.time()
        started_monotonic = time.monotonic()
        if not _GLOBAL_TOOL_SLOTS.acquire(blocking=False):
            finished = time.time()
            return ToolExecution(
                call=prepared.call,
                status=ToolStatus.CAPACITY_EXCEEDED,
                error="global tool execution capacity exhausted",
                risk=prepared.registered.spec.risk,
                approved=prepared.approved,
                started_at=started_wall,
                finished_at=finished,
                duration_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000),
            )
        try:
            try:
                output = prepared.registered.handler(**prepared.arguments)
                status = ToolStatus.SUCCESS
                error = None
            except Exception as exc:  # The execution is data for the agent loop to reason about.
                output = None
                status = ToolStatus.ERROR
                error = f"{type(exc).__name__}: {exc}"
        finally:
            _GLOBAL_TOOL_SLOTS.release()
        finished = time.time()
        return ToolExecution(
            call=prepared.call,
            status=status,
            output=output,
            error=error,
            risk=prepared.registered.spec.risk,
            approved=prepared.approved,
            started_at=started_wall,
            finished_at=finished,
            duration_ms=max(0.0, (time.monotonic() - started_monotonic) * 1000),
        )

    @staticmethod
    def _instant_result(
        call: ToolCall,
        status: ToolStatus,
        error: str,
        *,
        risk: RiskLevel | None = None,
        approved: bool | None = None,
    ) -> ToolExecution:
        now = time.time()
        return ToolExecution(
            call=call,
            status=status,
            error=error,
            risk=risk,
            approved=approved,
            started_at=now,
            finished_at=now,
        )
