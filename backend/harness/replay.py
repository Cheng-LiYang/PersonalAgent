"""Deterministic record/replay and fault-injection primitives for Agent runs.

The cassette format is deliberately small and dependency free.  A JSONL file
contains one ``run_cassette`` header followed by ordered ``event`` records.  All
data passes through the same redactor before it can reach memory or disk.

This module is intentionally independent from the Agent graph.  Graph nodes,
tool calls and model calls can therefore use the same harness without creating
an import cycle::

    with Recorder("run.jsonl", run_id="run-1") as recorder:
        recorder.record_call("tool.search", {"query": "RAG"}, {"hits": 2})

    with ReplaySession("run.jsonl") as replay:
        output = replay.replay("tool.search", {"query": "RAG"})
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final


CASSETTE_VERSION: Final = "1.0"
REDACTED: Final = "[REDACTED]"
_HEADER_TYPE: Final = "run_cassette"
_EVENT_TYPE: Final = "event"
_MISSING: Final = object()
_DEFAULT_CORRUPTION: Final = object()

_SENSITIVE_KEYS = frozenset(
    {
        "apikey",
        "token",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "authtoken",
        "authorization",
        "password",
        "passwd",
        "secret",
        "clientsecret",
        "secretkey",
        "privatekey",
        "cookie",
        "setcookie",
        "sessionid",
        "credential",
        "credentials",
    }
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_KEY_RE = re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{8,}\b")
_TENCENT_KEY_RE = re.compile(r"\bAKID[A-Za-z0-9]{8,}\b")
_CLOUD_KEY_RE = re.compile(
    r"\b(?:AKIA[0-9A-Z]{12,}|AIza[0-9A-Za-z_-]{20,}|gh[pousr]_[0-9A-Za-z]{20,}|xox[baprs]-[0-9A-Za-z-]{10,})\b"
)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b")
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&#\s]+"
)
_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[^,;\s]+"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----[\s\S]*?-----END [^-\r\n]*PRIVATE KEY-----"
)


class CassetteError(RuntimeError):
    """Base class for cassette failures."""


class CassetteSchemaError(CassetteError, ValueError):
    """Raised when a cassette does not satisfy the versioned JSONL schema."""


class RecorderClosedError(CassetteError):
    """Raised when recording after a recorder has been closed."""


class ReplayError(CassetteError):
    """Base class for deterministic replay failures."""


class MissingEventError(ReplayError):
    """Raised when the caller requests an event after the recording ended."""


class ReplayMismatchError(ReplayError):
    """Raised when the next kind or input differs from the recording."""


class ExtraEventsError(ReplayError):
    """Raised when replay completes with recorded events left unconsumed."""


class FaultPlanError(ValueError):
    """Raised when a fault rule is ambiguous or invalid."""


class InjectedFault(RuntimeError):
    """Default exception raised by an ``exception`` fault."""

    def __init__(self, message: str, *, kind: str, occurrence: int, rule_index: int) -> None:
        super().__init__(message)
        self.kind = kind
        self.occurrence = occurrence
        self.rule_index = rule_index


class InjectedTimeout(TimeoutError):
    """Timeout raised by a ``timeout`` fault without relying on wall-clock IO."""

    def __init__(self, message: str, *, kind: str, occurrence: int, rule_index: int) -> None:
        super().__init__(message)
        self.kind = kind
        self.occurrence = occurrence
        self.rule_index = rule_index


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_timestamp(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CassetteSchemaError(f"{path} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CassetteSchemaError(f"{path} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise CassetteSchemaError(f"{path} must include a timezone")
    return value


def _redact_text(value: str) -> str:
    if _PRIVATE_KEY_RE.search(value):
        return REDACTED
    value = _BEARER_RE.sub("Bearer " + REDACTED, value)
    value = _KEY_RE.sub(REDACTED, value)
    value = _TENCENT_KEY_RE.sub(REDACTED, value)
    value = _CLOUD_KEY_RE.sub(REDACTED, value)
    value = _JWT_RE.sub(REDACTED, value)
    value = _QUERY_SECRET_RE.sub(lambda match: match.group(1) + REDACTED, value)
    value = _ASSIGNMENT_SECRET_RE.sub(lambda match: match.group(1) + "=" + REDACTED, value)
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        (
            "apikey",
            "accesstoken",
            "refreshtoken",
            "idtoken",
            "authtoken",
            "clientsecret",
            "secretkey",
            "privatekey",
            "password",
            "passwd",
            "credentials",
        )
    )


def redact(value: Any, *, _path: str = "$", _depth: int = 0) -> Any:
    """Return a JSON-safe, deeply copied value with credentials removed.

    Unsupported objects are rejected instead of being converted with ``repr``;
    object representations frequently contain connection strings or tokens.
    """

    if _depth > 64:
        raise TypeError(f"{_path} exceeds the maximum payload nesting depth")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{_path} contains a non-finite number")
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{_path} contains a non-string object key")
            child_path = f"{_path}.{key}"
            result[key] = REDACTED if _is_sensitive_key(key) else redact(
                item, _path=child_path, _depth=_depth + 1
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            redact(item, _path=f"{_path}[{index}]", _depth=_depth + 1)
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{_path} contains unsupported type {type(value).__name__}")


def _require_exact_keys(
    value: Mapping[str, Any], required: set[str], *, path: str
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing or extra:
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unknown " + ", ".join(extra))
        raise CassetteSchemaError(f"{path} has invalid fields: {'; '.join(detail)}")


def _json_line(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CassetteSchemaError("cassette contains a non-JSON value") from exc


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CassetteSchemaError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class Event:
    """One immutable, ordered observation in a run cassette."""

    sequence: int
    timestamp: str
    kind: str
    payload: Mapping[str, Any]
    version: str = CASSETTE_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise CassetteSchemaError("event.sequence must be a positive integer")
        _validate_timestamp(self.timestamp, path="event.timestamp")
        if not isinstance(self.kind, str) or not self.kind.strip() or len(self.kind) > 128:
            raise CassetteSchemaError("event.kind must be a non-empty string of at most 128 characters")
        if self.version != CASSETTE_VERSION:
            raise CassetteSchemaError(
                f"unsupported event version {self.version!r}; expected {CASSETTE_VERSION!r}"
            )
        if not isinstance(self.payload, Mapping):
            raise CassetteSchemaError("event.payload must be an object")
        safe_payload = redact(self.payload, _path="event.payload")
        object.__setattr__(self, "payload", MappingProxyType(safe_payload))

    @classmethod
    def create(
        cls,
        sequence: int,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timestamp: str | None = None,
    ) -> "Event":
        return cls(sequence, timestamp or _utc_now(), kind, payload or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": _EVENT_TYPE,
            "version": self.version,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "payload": copy.deepcopy(dict(self.payload)),
        }

    @classmethod
    def from_dict(cls, value: Any, *, line_number: int | None = None) -> "Event":
        path = f"line {line_number}" if line_number is not None else "event"
        if not isinstance(value, Mapping):
            raise CassetteSchemaError(f"{path} must be a JSON object")
        required = {"record_type", "version", "sequence", "timestamp", "kind", "payload"}
        _require_exact_keys(value, required, path=path)
        if value["record_type"] != _EVENT_TYPE:
            raise CassetteSchemaError(f"{path}.record_type must be {_EVENT_TYPE!r}")
        try:
            return cls(
                sequence=value["sequence"],
                timestamp=value["timestamp"],
                kind=value["kind"],
                payload=value["payload"],
                version=value["version"],
            )
        except (CassetteSchemaError, TypeError) as exc:
            raise CassetteSchemaError(f"{path}: {exc}") from exc


# A descriptive alias for callers that prefer an explicit domain name.
RunEvent = Event


class RunCassette:
    """Thread-safe collection of events with strict JSONL persistence."""

    def __init__(
        self,
        run_id: str | None = None,
        *,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        events: Iterable[Event] = (),
        version: str = CASSETTE_VERSION,
    ) -> None:
        if version != CASSETTE_VERSION:
            raise CassetteSchemaError(
                f"unsupported cassette version {version!r}; expected {CASSETTE_VERSION!r}"
            )
        self.run_id = run_id or str(uuid.uuid4())
        if not isinstance(self.run_id, str) or not self.run_id.strip() or len(self.run_id) > 256:
            raise CassetteSchemaError("run_id must be a non-empty string of at most 256 characters")
        self.created_at = _validate_timestamp(created_at or _utc_now(), path="created_at")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise CassetteSchemaError("metadata must be an object")
        self.metadata: Mapping[str, Any] = MappingProxyType(
            redact(metadata or {}, _path="metadata")
        )
        self.version = version
        self._lock = threading.RLock()
        self._events: list[Event] = []
        for event in events:
            self._append_loaded(event)

    @property
    def events(self) -> tuple[Event, ...]:
        with self._lock:
            return tuple(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def _append_loaded(self, event: Event) -> None:
        if not isinstance(event, Event):
            raise CassetteSchemaError("events must contain Event instances")
        expected = len(self._events) + 1
        if event.sequence != expected:
            raise CassetteSchemaError(
                f"event sequence must be contiguous: expected {expected}, got {event.sequence}"
            )
        self._events.append(event)

    def append(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timestamp: str | None = None,
    ) -> Event:
        with self._lock:
            event = Event.create(len(self._events) + 1, kind, payload, timestamp=timestamp)
            self._events.append(event)
            return event

    def _header(self) -> dict[str, Any]:
        return {
            "record_type": _HEADER_TYPE,
            "version": self.version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }

    def to_jsonl(self) -> str:
        with self._lock:
            records = [self._header(), *(event.to_dict() for event in self._events)]
        return "\n".join(_json_line(record) for record in records) + "\n"

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically replace ``path`` with a complete, fsynced snapshot."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_jsonl()
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    @classmethod
    def from_jsonl(cls, content: str) -> "RunCassette":
        if not isinstance(content, str):
            raise CassetteSchemaError("cassette content must be text")
        lines = content.splitlines()
        if not lines:
            raise CassetteSchemaError("cassette is empty")
        parsed: list[Any] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise CassetteSchemaError(f"line {line_number} must not be blank")
            try:
                parsed.append(json.loads(line, object_pairs_hook=_strict_json_object))
            except json.JSONDecodeError as exc:
                raise CassetteSchemaError(f"line {line_number} is not valid JSON: {exc.msg}") from exc
            except CassetteSchemaError as exc:
                raise CassetteSchemaError(f"line {line_number}: {exc}") from exc

        header = parsed[0]
        if not isinstance(header, Mapping):
            raise CassetteSchemaError("line 1 must be a JSON object")
        required = {"record_type", "version", "run_id", "created_at", "metadata"}
        _require_exact_keys(header, required, path="line 1")
        if header["record_type"] != _HEADER_TYPE:
            raise CassetteSchemaError(f"line 1.record_type must be {_HEADER_TYPE!r}")
        if not isinstance(header["metadata"], Mapping):
            raise CassetteSchemaError("line 1.metadata must be an object")

        events = [
            Event.from_dict(record, line_number=line_number)
            for line_number, record in enumerate(parsed[1:], start=2)
        ]
        try:
            return cls(
                run_id=header["run_id"],
                created_at=header["created_at"],
                metadata=header["metadata"],
                events=events,
                version=header["version"],
            )
        except (CassetteSchemaError, TypeError) as exc:
            raise CassetteSchemaError(f"line 1: {exc}") from exc

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "RunCassette":
        try:
            content = Path(path).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise CassetteSchemaError("cassette must be UTF-8 text") from exc
        return cls.from_jsonl(content)


class Recorder:
    """Durably record events, atomically flushing after every call by default."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        flush_each_event: bool = True,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"cassette already exists: {self.path}")
        self.cassette = RunCassette(run_id, metadata=metadata)
        self.flush_each_event = bool(flush_each_event)
        self._lock = threading.RLock()
        self._closed = False

    def record(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timestamp: str | None = None,
    ) -> Event:
        with self._lock:
            if self._closed:
                raise RecorderClosedError("recorder is closed")
            event = self.cassette.append(kind, payload, timestamp=timestamp)
            if self.flush_each_event:
                self.cassette.save(self.path)
            return event

    def record_call(
        self,
        kind: str,
        input_payload: Any,
        output_payload: Any,
        *,
        timestamp: str | None = None,
    ) -> Event:
        return self.record(
            kind,
            {"input": redact(input_payload, _path="input"), "output": redact(output_payload, _path="output")},
            timestamp=timestamp,
        )

    def flush(self) -> None:
        with self._lock:
            self.cassette.save(self.path)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self.cassette.save(self.path)
                self._closed = True

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class ReplaySession:
    """Consume a cassette exactly once and in the recorded order."""

    def __init__(
        self,
        source: RunCassette | str | os.PathLike[str],
        *,
        verify_on_close: bool = True,
    ) -> None:
        self.cassette = source if isinstance(source, RunCassette) else RunCassette.load(source)
        self.verify_on_close = verify_on_close
        self._cursor = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self.cassette) - self._cursor

    @property
    def complete(self) -> bool:
        return self.remaining == 0

    def consume(self, kind: str, input_payload: Any = _MISSING) -> Event:
        """Consume the next event, optionally checking its recorded input.

        A failed comparison never advances the cursor, making failures
        inspectable and deterministic.
        """

        with self._lock:
            if self._closed:
                raise ReplayError("replay session is closed")
            events = self.cassette.events
            if self._cursor >= len(events):
                raise MissingEventError(
                    f"no event remains while consuming kind {kind!r} at position {self._cursor + 1}"
                )
            event = events[self._cursor]
            if event.kind != kind:
                raise ReplayMismatchError(
                    f"event {event.sequence} kind mismatch: expected {event.kind!r}, got {kind!r}"
                )
            if input_payload is not _MISSING:
                actual = redact(input_payload, _path="replay.input")
                recorded = event.payload.get("input", event.payload)
                if actual != recorded:
                    raise ReplayMismatchError(
                        f"event {event.sequence} input mismatch for kind {kind!r}: "
                        f"expected {_json_line({'input': recorded})}, "
                        f"got {_json_line({'input': actual})}"
                    )
            self._cursor += 1
            return event

    # ``next_event`` reads naturally at graph integration sites.
    next_event = consume

    def replay(self, kind: str, input_payload: Any = _MISSING) -> Any:
        event = self.consume(kind, input_payload)
        value = event.payload.get("output", event.payload)
        return copy.deepcopy(value)

    def assert_complete(self) -> None:
        with self._lock:
            remaining = len(self.cassette) - self._cursor
            if remaining:
                next_event = self.cassette.events[self._cursor]
                raise ExtraEventsError(
                    f"replay has {remaining} unconsumed event(s); "
                    f"next is {next_event.kind!r} at sequence {next_event.sequence}"
                )

    verify_complete = assert_complete

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self.verify_on_close:
                self.assert_complete()
            self._closed = True

    def __enter__(self) -> "ReplaySession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        # Do not hide a business exception with an unrelated extra-event error.
        if exc_type is None:
            self.close()


@dataclass(frozen=True, slots=True)
class FaultRule:
    """Inject ``action`` at the Nth matching event (and optionally thereafter)."""

    kind: str
    occurrence: int = 1
    action: str = "exception"
    max_hits: int = 1
    repeat_every: int | None = None
    probability: float = 1.0
    delay_seconds: float = 0.0
    message: str = "fault injected by harness"
    replacement: Any = field(default=_DEFAULT_CORRUPTION, repr=False, compare=False)
    exception_type: type[Exception] = field(default=InjectedFault, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise FaultPlanError("fault kind must be a non-empty string")
        if isinstance(self.occurrence, bool) or not isinstance(self.occurrence, int) or self.occurrence < 1:
            raise FaultPlanError("occurrence must be a positive integer")
        if self.action not in {"delay", "exception", "timeout", "corrupt"}:
            raise FaultPlanError("action must be delay, exception, timeout, or corrupt")
        if isinstance(self.max_hits, bool) or not isinstance(self.max_hits, int) or self.max_hits < 1:
            raise FaultPlanError("max_hits must be a positive integer")
        if self.repeat_every is not None and (
            isinstance(self.repeat_every, bool)
            or not isinstance(self.repeat_every, int)
            or self.repeat_every < 1
        ):
            raise FaultPlanError("repeat_every must be a positive integer or None")
        if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
            raise FaultPlanError("probability must be numeric")
        if not 0.0 <= float(self.probability) <= 1.0:
            raise FaultPlanError("probability must be between 0 and 1")
        if isinstance(self.delay_seconds, bool) or not isinstance(self.delay_seconds, (int, float)):
            raise FaultPlanError("delay_seconds must be numeric")
        if not math.isfinite(float(self.delay_seconds)) or self.delay_seconds < 0:
            raise FaultPlanError("delay_seconds must be finite and non-negative")
        if not isinstance(self.message, str):
            raise FaultPlanError("message must be a string")
        if not isinstance(self.exception_type, type) or not issubclass(self.exception_type, Exception):
            raise FaultPlanError("exception_type must be an Exception class")


@dataclass(frozen=True, slots=True)
class FaultPlan:
    """Immutable set of ordered rules and its deterministic random seed."""

    rules: tuple[FaultRule, ...] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        normalized: list[FaultRule] = []
        for rule in self.rules:
            if isinstance(rule, FaultRule):
                normalized.append(rule)
            elif isinstance(rule, Mapping):
                try:
                    normalized.append(FaultRule(**dict(rule)))
                except TypeError as exc:
                    raise FaultPlanError(f"invalid fault rule: {exc}") from exc
            else:
                raise FaultPlanError("rules must contain FaultRule objects or mappings")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise FaultPlanError("seed must be an integer")
        object.__setattr__(self, "rules", tuple(normalized))

    @classmethod
    def single(
        cls,
        kind: str,
        *,
        occurrence: int = 1,
        action: str = "exception",
        seed: int = 0,
        **options: Any,
    ) -> "FaultPlan":
        return cls((FaultRule(kind, occurrence, action, **options),), seed=seed)


@dataclass(frozen=True, slots=True)
class FaultHit:
    rule_index: int
    kind: str
    occurrence: int
    action: str
    hit_number: int


class FaultInjector:
    """Apply a fault plan with thread-safe counters and deterministic RNG."""

    def __init__(
        self,
        plan: FaultPlan | Iterable[FaultRule],
        *,
        seed: int | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not isinstance(plan, FaultPlan):
            plan = FaultPlan(tuple(plan), seed=0 if seed is None else seed)
        elif seed is not None:
            plan = FaultPlan(plan.rules, seed=seed)
        if not callable(sleep):
            raise FaultPlanError("sleep must be callable")
        self.plan = plan
        self._sleep = sleep
        self._random = random.Random(plan.seed)
        self._lock = threading.RLock()
        self._occurrences: dict[str, int] = {}
        self._rule_hits = [0 for _ in plan.rules]
        self._history: list[FaultHit] = []

    @property
    def hit_count(self) -> int:
        with self._lock:
            return sum(self._rule_hits)

    @property
    def hit_counts(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._rule_hits)

    @property
    def history(self) -> tuple[FaultHit, ...]:
        with self._lock:
            return tuple(self._history)

    def occurrence_count(self, kind: str) -> int:
        with self._lock:
            return self._occurrences.get(kind, 0)

    def reset(self) -> None:
        with self._lock:
            self._occurrences.clear()
            self._rule_hits = [0 for _ in self.plan.rules]
            self._history.clear()
            self._random.seed(self.plan.seed)

    def _matches(self, rule: FaultRule, kind: str, occurrence: int, index: int) -> bool:
        if rule.kind not in {kind, "*"} or self._rule_hits[index] >= rule.max_hits:
            return False
        if occurrence < rule.occurrence:
            return False
        if rule.repeat_every is None:
            if occurrence != rule.occurrence:
                return False
        elif (occurrence - rule.occurrence) % rule.repeat_every != 0:
            return False
        probability = float(rule.probability)
        return probability >= 1.0 or (probability > 0.0 and self._random.random() < probability)

    def inject(self, kind: str, payload: Any = None) -> Any:
        """Run matching hooks and return the possibly corrupted payload.

        Delay and timeout waiting happens outside the state lock.  The matching
        decision and seeded random draws remain serialized, so a given call
        order always produces the same fault history.
        """

        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("kind must be a non-empty string")
        with self._lock:
            occurrence = self._occurrences.get(kind, 0) + 1
            self._occurrences[kind] = occurrence
            matches: list[tuple[int, FaultRule, FaultHit, int]] = []
            for index, rule in enumerate(self.plan.rules):
                if self._matches(rule, kind, occurrence, index):
                    self._rule_hits[index] += 1
                    hit = FaultHit(index, kind, occurrence, rule.action, self._rule_hits[index])
                    self._history.append(hit)
                    # Draw corruption entropy while holding the RNG lock.
                    entropy = self._random.getrandbits(64)
                    matches.append((index, rule, hit, entropy))

        result = payload
        for index, rule, hit, entropy in matches:
            if rule.action == "delay":
                self._sleep(float(rule.delay_seconds))
            elif rule.action == "timeout":
                if rule.delay_seconds:
                    self._sleep(float(rule.delay_seconds))
                raise InjectedTimeout(
                    rule.message, kind=kind, occurrence=occurrence, rule_index=index
                )
            elif rule.action == "exception":
                if rule.delay_seconds:
                    self._sleep(float(rule.delay_seconds))
                if rule.exception_type is InjectedFault:
                    raise InjectedFault(
                        rule.message, kind=kind, occurrence=occurrence, rule_index=index
                    )
                raise rule.exception_type(rule.message)
            else:
                result = _corrupt(result, rule.replacement, entropy)
        return result

    # Alias used by Agent nodes before invoking an external dependency.
    before = inject


def _corrupt(value: Any, replacement: Any, entropy: int) -> Any:
    if replacement is not _DEFAULT_CORRUPTION:
        return copy.deepcopy(replacement)
    result = copy.deepcopy(value)
    rng = random.Random(entropy)
    if isinstance(result, dict):
        if not result:
            result["__corrupted__"] = True
        else:
            keys = sorted(result, key=lambda item: str(item))
            key = keys[rng.randrange(len(keys))]
            result[key] = _corrupt(result[key], _DEFAULT_CORRUPTION, rng.getrandbits(64))
        return result
    if isinstance(result, list):
        if not result:
            return [{"__corrupted__": True}]
        index = rng.randrange(len(result))
        result[index] = _corrupt(result[index], _DEFAULT_CORRUPTION, rng.getrandbits(64))
        return result
    if isinstance(result, str):
        if not result:
            return "<corrupted>"
        index = rng.randrange(len(result))
        replacement_character = "?" if result[index] != "?" else "!"
        return result[:index] + replacement_character + result[index + 1 :]
    if isinstance(result, bytes):
        if not result:
            return b"\x00"
        mutable = bytearray(result)
        index = rng.randrange(len(mutable))
        mutable[index] ^= 0xFF
        return bytes(mutable)
    if isinstance(result, bool):
        return not result
    if isinstance(result, (int, float)):
        return result + 1
    return {"__corrupted__": True, "original_type": type(result).__name__}


__all__ = [
    "CASSETTE_VERSION",
    "REDACTED",
    "CassetteError",
    "CassetteSchemaError",
    "RecorderClosedError",
    "ReplayError",
    "MissingEventError",
    "ReplayMismatchError",
    "ExtraEventsError",
    "FaultPlanError",
    "InjectedFault",
    "InjectedTimeout",
    "Event",
    "RunEvent",
    "RunCassette",
    "Recorder",
    "ReplaySession",
    "FaultRule",
    "FaultPlan",
    "FaultHit",
    "FaultInjector",
    "redact",
]
