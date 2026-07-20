"""Shared evidence helpers for allocation application benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

ALLOCATOR_ID = "timed-application-engine"


def validate_inputs(operations: int, seed: int) -> None:
    """Validate the uniform allocation benchmark inputs before engine setup."""
    if isinstance(operations, bool) or not isinstance(operations, int):
        raise TypeError("operations must be an integer")
    if operations <= 0:
        raise ValueError("operations must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")


def stable_evidence(value: Any) -> Any:
    """Preserve public evidence while normalizing per-instance allocator IDs."""
    if is_dataclass(value) and not isinstance(value, type):
        return stable_evidence(asdict(value))
    if isinstance(value, Mapping):
        return {
            key: ALLOCATOR_ID if key == "allocator_id" else stable_evidence(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(stable_evidence(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [stable_evidence(item) for item in value]
    return value


def span(start: int, end: int) -> dict[str, int]:
    """Return canonical span evidence without constructing a production value."""
    return {"start": start, "end": end}


def handle(
    allocation_id: int, owner: object, start: int, end: int
) -> dict[str, object]:
    """Return normalized public allocation-handle evidence."""
    return {
        "allocator_id": ALLOCATOR_ID,
        "allocation_id": allocation_id,
        "owner": owner,
        "span": span(start, end),
    }


def empty_fragmentation(
    total_space: int, *, reserved_space: int = 0
) -> dict[str, int | float]:
    """Return diagnostics for one fully coalesced free application domain."""
    total_free = total_space - reserved_space
    return {
        "total_space": total_space,
        "allocated_space": 0,
        "reserved_space": reserved_space,
        "total_free": total_free,
        "free_chunks": 1,
        "largest_free_chunk": total_free,
        "average_free_chunk": float(total_free),
        "fragmentation": 0.0,
    }
