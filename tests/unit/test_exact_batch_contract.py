"""Immutable stable exact-batch API and native ABI contract for 1.1."""

from __future__ import annotations

import importlib
import struct
from dataclasses import fields

import pytest

import treemendous
import treemendous.exact_batch as exact_batch
from treemendous.exact_batch import (
    BatchLimits,
    BatchMutation,
    ExactBatchRangeSet,
    MutationOpcode,
)

_EXPECTED_PUBLIC_NAMES = [
    "BatchLimitError",
    "BatchLimits",
    "BatchMutation",
    "ExactBatchRangeSet",
    "MutationOpcode",
    "PackedMutationResults",
]
_EXPECTED_LIMITS = {
    "max_operations": 1_000_000,
    "max_live_intervals": 100_000,
    "max_changed_spans": 2_000_000,
    "max_result_bytes": 256 * 1024 * 1024,
    "max_work_units": 100_000_000,
}


def test_stable_exact_batch_api_and_abi_contract_is_immutable() -> None:
    assert exact_batch.__all__ == _EXPECTED_PUBLIC_NAMES
    assert tuple(MutationOpcode) == (
        MutationOpcode.ADD,
        MutationOpcode.DISCARD,
        MutationOpcode.DISCARD_REQUIRE_COVERED,
    )
    assert [opcode.value for opcode in MutationOpcode] == [0, 1, 2]

    limits = BatchLimits()
    assert [field.name for field in fields(BatchLimits)] == list(_EXPECTED_LIMITS)
    assert {
        field.name: getattr(limits, field.name) for field in fields(BatchLimits)
    } == _EXPECTED_LIMITS

    row = struct.Struct("@qqq")
    assert row.size == 24
    assert row.unpack(row.pack(2, -7, 11)) == (2, -7, 11)

    ranges = ExactBatchRangeSet((0, 8), initially_available=False, limits=limits)
    assert ranges.limits is limits
    with pytest.raises(AttributeError):
        ranges.limits = BatchLimits()  # type: ignore[misc]

    packed_results = ranges.mutate_packed(
        b"".join((row.pack(0, 1, 5), row.pack(1, 2, 4)))
    )
    assert len(packed_results) == 2
    assert packed_results.changed_offsets.format == "Q"
    assert packed_results.changed_offsets.shape == (3,)
    assert packed_results.changed_offsets.tolist() == [0, 1, 2]
    assert packed_results.changed_spans.format == "q"
    assert packed_results.changed_spans.shape == (2, 2)
    assert packed_results.changed_spans.tolist() == [[1, 5], [2, 4]]
    assert packed_results.changed_lengths.format == "q"
    assert packed_results.changed_lengths.shape == (2,)
    assert packed_results.changed_lengths.tolist() == [4, 2]
    assert packed_results.fully_covered.format == "B"
    assert packed_results.fully_covered.shape == (2,)
    assert packed_results.fully_covered.tolist() == [0, 1]
    assert all(
        view.readonly
        for view in (
            packed_results.changed_offsets,
            packed_results.changed_spans,
            packed_results.changed_lengths,
            packed_results.fully_covered,
        )
    )
    assert (
        packed_results.changed_offsets.nbytes,
        packed_results.changed_spans.nbytes,
        packed_results.changed_lengths.nbytes,
        packed_results.fully_covered.nbytes,
    ) == (24, 32, 16, 2)

    assert not set(_EXPECTED_PUBLIC_NAMES) & set(treemendous.__all__)
    assert all(not hasattr(treemendous, name) for name in _EXPECTED_PUBLIC_NAMES)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("treemendous.experimental")

    from treemendous.backends.types import Capability
    from treemendous.protocols import RangeSetProtocol

    assert "EXACT_BATCH" not in Capability.__members__
    assert "mutate" not in RangeSetProtocol.__dict__
    assert "mutate_packed" not in RangeSetProtocol.__dict__


def test_ergonomic_row_layout_matches_packed_contract() -> None:
    mutation = BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, -7, 11)
    assert (int(mutation.opcode), mutation.start, mutation.end) == (2, -7, 11)
    assert struct.pack("@qqq", int(mutation.opcode), mutation.start, mutation.end) == (
        struct.pack("@qqq", 2, -7, 11)
    )
