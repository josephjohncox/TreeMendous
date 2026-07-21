"""Stable exact, ordered, whole-batch-atomic CPU geometry mutations.

This specialized API is intentionally separate from the root ``RangeSet`` API and
backend registry.  It manages geometry only: payloads, allocation, and generic
queries are not supported.
"""

from __future__ import annotations

import struct
import sys
from collections.abc import Iterable
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Final

from treemendous.cpp._exact_batch import (
    BatchLimitError,
    PackedMutationResults,
)
from treemendous.cpp._exact_batch import (
    ExactBatchManager as _ExactBatchManager,
)
from treemendous.domain import (
    DomainInput,
    IntervalResult,
    ManagedDomain,
    MutationResult,
    RangeSnapshot,
    validate_coordinate,
)

_INT64_MIN: Final = -(1 << 63)
_INT64_MAX: Final = (1 << 63) - 1
_OPERATION: Final = struct.Struct("@qqq")


class MutationOpcode(IntEnum):
    """Native ABI opcode stored in the first signed-int64 field of each row."""

    ADD = 0
    DISCARD = 1
    DISCARD_REQUIRE_COVERED = 2


@dataclass(frozen=True, slots=True)
class BatchMutation:
    """One validated half-open geometry mutation in an ordered batch."""

    opcode: MutationOpcode | int
    start: int
    end: int

    def __post_init__(self) -> None:
        raw_opcode = self.opcode
        if isinstance(raw_opcode, bool) or not isinstance(raw_opcode, int):
            raise TypeError("opcode must be a MutationOpcode or integer")
        opcode = MutationOpcode(raw_opcode)
        start = validate_coordinate(self.start, "start")
        end = validate_coordinate(self.end, "end")
        if not _INT64_MIN <= start <= _INT64_MAX:
            raise OverflowError("start is outside signed int64")
        if not _INT64_MIN <= end <= _INT64_MAX:
            raise OverflowError("end is outside signed int64")
        if start >= end:
            raise ValueError("span must satisfy start < end")
        object.__setattr__(self, "opcode", opcode)


@dataclass(frozen=True, slots=True)
class BatchLimits:
    """Per-instance resource limits for exact batch staging.

    Defaults are deliberately conservative for production use. ``max_work_units``
    counts one row dispatch plus the number of live intervals presented to each
    row; it bounds the sorted-vector staging cost rather than elapsed time.
    """

    max_operations: int = 1_000_000
    max_live_intervals: int = 100_000
    max_changed_spans: int = 2_000_000
    max_result_bytes: int = 256 * 1024 * 1024
    max_work_units: int = 100_000_000

    def __post_init__(self) -> None:
        for definition in fields(self):
            value = getattr(self, definition.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{definition.name} must be an integer")
            if value <= 0:
                raise ValueError(f"{definition.name} must be greater than zero")
            if value > sys.maxsize:
                raise OverflowError(f"{definition.name} exceeds signed size range")


class ExactBatchRangeSet:
    """Independent sorted-vector geometry specialized for exact batch mutation.

    Args:
        domain: A non-empty managed domain, normalized with stable package rules.
        initially_available: Whether every managed component starts available.
        limits: Checked per-instance staging and packed-result limits.
    """

    def __init__(
        self,
        domain: DomainInput,
        *,
        initially_available: bool = True,
        limits: BatchLimits | None = None,
    ) -> None:
        if not isinstance(initially_available, bool):
            raise TypeError("initially_available must be a bool")
        if limits is None:
            limits = BatchLimits()
        elif not isinstance(limits, BatchLimits):
            raise TypeError("limits must be a BatchLimits instance")
        normalized = (
            domain if isinstance(domain, ManagedDomain) else ManagedDomain(domain)
        )
        for span in normalized.spans:
            if not (_INT64_MIN <= span.start <= _INT64_MAX):
                raise OverflowError("managed domain start is outside signed int64")
            if not (_INT64_MIN <= span.end <= _INT64_MAX):
                raise OverflowError("managed domain end is outside signed int64")
        self._domain = normalized
        self._limits = limits
        self._manager = _ExactBatchManager(
            [(span.start, span.end) for span in normalized.spans],
            initially_available,
            limits.max_operations,
            limits.max_live_intervals,
            limits.max_changed_spans,
            limits.max_result_bytes,
            limits.max_work_units,
        )

    @property
    def domain(self) -> ManagedDomain:
        """Return the normalized immutable managed domain."""
        return self._domain

    def mutate(self, operations: Iterable[BatchMutation]) -> tuple[MutationResult, ...]:
        """Validate and apply an ergonomic ordered batch atomically."""
        try:
            iterator = iter(operations)
        except TypeError:
            raise TypeError("operations must be an iterable of BatchMutation") from None
        storage = bytearray()
        count = 0
        for operation in iterator:
            if not isinstance(operation, BatchMutation):
                raise TypeError("operations must contain only BatchMutation values")
            if count >= self._limits.max_operations:
                raise BatchLimitError("max_operations limit exceeded")
            storage.extend(
                _OPERATION.pack(int(operation.opcode), operation.start, operation.end)
            )
            count += 1
        return self.mutate_packed(bytes(storage)).materialize()

    def mutate_packed(self, operations: bytes) -> PackedMutationResults:
        """Apply native-endian signed-int64 triples from exact immutable ``bytes``.

        Every 24-byte row is ``(opcode, start, end)``. No other buffer exporter,
        including ``bytearray`` and ``memoryview``, is accepted.
        """
        if type(operations) is not bytes:
            raise TypeError("operations must be exact immutable bytes")
        return self._manager.mutate_packed(operations)

    def snapshot(self) -> RangeSnapshot:
        """Return one exact canonical pre- or post-batch state publication."""
        raw_intervals, total = self._manager.snapshot_data()
        return RangeSnapshot(
            tuple(IntervalResult(start, end) for start, end in raw_intervals),
            total,
            self._domain,
        )


__all__ = [
    "BatchLimitError",
    "BatchLimits",
    "BatchMutation",
    "ExactBatchRangeSet",
    "MutationOpcode",
    "PackedMutationResults",
]
