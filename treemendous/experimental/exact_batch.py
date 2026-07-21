"""Experimental exact, ordered, whole-batch-atomic CPU mutations.

This module is deliberately outside the stable backend and protocol surfaces.  The
implementation owns independent geometry and accepts only packed signed-int64
operation buffers; it does not implement payloads or general range queries.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final

from treemendous.cpp._exact_batch import (
    ExactBatchManager as _ExactBatchManager,
)
from treemendous.cpp._exact_batch import (
    PackedMutationResults,
)
from treemendous.domain import DomainInput, IntervalResult, ManagedDomain, RangeSnapshot

_INT64_MIN: Final = -(1 << 63)
_INT64_MAX: Final = (1 << 63) - 1


class MutationOpcode(IntEnum):
    """Opcode values stored in the first signed-int64 column of each row."""

    ADD = 0
    DISCARD = 1
    DISCARD_REQUIRE_COVERED = 2


class ExactBatchRangeSet:
    """Independent sorted-vector geometry specialized for exact batch mutation.

    Args:
        domain: A non-empty managed domain, normalized with stable package rules.
        initially_available: Whether every managed component starts available.
    """

    def __init__(
        self, domain: DomainInput, *, initially_available: bool = True
    ) -> None:
        if not isinstance(initially_available, bool):
            raise TypeError("initially_available must be a bool")
        normalized = (
            domain if isinstance(domain, ManagedDomain) else ManagedDomain(domain)
        )
        for span in normalized.spans:
            if not (_INT64_MIN <= span.start <= _INT64_MAX):
                raise OverflowError("managed domain start is outside signed int64")
            if not (_INT64_MIN <= span.end <= _INT64_MAX):
                raise OverflowError("managed domain end is outside signed int64")
        self._domain = normalized
        self._manager = _ExactBatchManager(
            [(span.start, span.end) for span in normalized.spans],
            initially_available,
        )

    @property
    def domain(self) -> ManagedDomain:
        """Return the normalized immutable managed domain."""
        return self._domain

    def mutate_packed(self, operations: object) -> PackedMutationResults:
        """Apply an ordered packed trace atomically and return packed exact deltas.

        ``operations`` must export a C-contiguous, native-endian, signed-int64
        buffer whose shape is either ``(3*N,)`` or ``(N, 3)``.  Columns are
        ``opcode, start, end``.  Tuple/list hot inputs are intentionally rejected.
        """
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
    "ExactBatchRangeSet",
    "MutationOpcode",
    "PackedMutationResults",
]
