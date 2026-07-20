"""Correctness-first timing harness for concrete application engines."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter_ns
from typing import Any, TypeVar
from uuid import UUID

RawResult = TypeVar("RawResult")
CanonicalValue = (
    None
    | bool
    | int
    | float
    | str
    | list["CanonicalValue"]
    | dict[str, "CanonicalValue"]
)


@dataclass(frozen=True)
class ApplicationOutcome:
    """Canonical result evidence and complete state from one execution."""

    results: Any
    final_state: Any
    counters: Any


@dataclass(frozen=True)
class ApplicationSample:
    """Validated timing evidence for one concrete scenario engine."""

    scenario_id: str
    operations: int
    execution_ns: int
    result_checksum: str
    state_checksum: str
    counters_checksum: str
    evidence_checksum: str
    validated: bool = True

    def to_dict(self) -> dict[str, int | str | bool]:
        """Return stable JSON artifact data."""
        return {
            "scenario_id": self.scenario_id,
            "operations": self.operations,
            "execution_ns": self.execution_ns,
            "result_checksum": self.result_checksum,
            "state_checksum": self.state_checksum,
            "counters_checksum": self.counters_checksum,
            "evidence_checksum": self.evidence_checksum,
            "validated": self.validated,
        }


def canonicalize(value: Any) -> CanonicalValue:
    """Convert bounded evidence to an unambiguous JSON-compatible value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("application evidence floats must be finite")
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Path):
        return {"path": value.as_posix()}
    if isinstance(value, UUID):
        return {"uuid": str(value)}
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize(asdict(value))
    if isinstance(value, Mapping):
        entries: list[CanonicalValue] = [
            [canonicalize(key), canonicalize(item)] for key, item in value.items()
        ]
        entries.sort(key=_encoded)
        return {"mapping": entries}
    if isinstance(value, Set):
        items = [canonicalize(item) for item in value]
        items.sort(key=_encoded)
        return {"set": items}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [canonicalize(item) for item in value]
    raise TypeError(f"unsupported application evidence type: {type(value).__name__}")


def _encoded(value: CanonicalValue) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _is_uuid_string(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _normalize_uuid_relations(value: CanonicalValue) -> CanonicalValue:
    """Replace UUID values by first-occurrence aliases for reproducible hashes."""
    aliases: dict[str, str] = {}

    def normalize(item: CanonicalValue) -> CanonicalValue:
        if isinstance(item, str) and _is_uuid_string(item):
            return aliases.setdefault(item, f"uuid-{len(aliases)}")
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        return item

    return normalize(value)


def evidence_checksum(value: Any) -> str:
    """Hash evidence while retaining UUID equality and distinctness relations."""
    canonical = canonicalize(value)
    encoded = (_encoded(_normalize_uuid_relations(canonical)) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def run_application_case(
    *,
    scenario_id: str,
    operations: int,
    execute: Callable[[], RawResult],
    observe: Callable[[RawResult], ApplicationOutcome],
    oracle: Callable[[], ApplicationOutcome],
    timer: Callable[[], int] = perf_counter_ns,
) -> ApplicationSample:
    """Time execution, then validate that same instance against an oracle.

    ``execute`` alone is inside the timing interval. ``observe`` must inspect the
    instance closed over by ``execute``; constructing or replaying a fresh
    application there would violate the contract. Canonicalization, checksums,
    and all independent oracle work occur after the end timestamp.
    """
    if not scenario_id:
        raise ValueError("scenario_id must be nonempty")
    if isinstance(operations, bool) or not isinstance(operations, int):
        raise TypeError("operations must be an integer")
    if operations <= 0:
        raise ValueError("operations must be positive")

    started = timer()
    raw = execute()
    finished = timer()
    if finished < started:
        raise ValueError("benchmark timer moved backwards")

    outcome = observe(raw)
    actual = canonicalize(outcome)
    expected = canonicalize(oracle())
    if actual != expected:
        raise AssertionError(
            f"application benchmark evidence differs for {scenario_id}: "
            f"actual={actual!r}, expected={expected!r}"
        )
    if not isinstance(actual, dict):
        raise TypeError("canonical application outcome must be a mapping")

    return ApplicationSample(
        scenario_id=scenario_id,
        operations=operations,
        execution_ns=finished - started,
        result_checksum=evidence_checksum(outcome.results),
        state_checksum=evidence_checksum(outcome.final_state),
        counters_checksum=evidence_checksum(outcome.counters),
        evidence_checksum=evidence_checksum(outcome),
    )
