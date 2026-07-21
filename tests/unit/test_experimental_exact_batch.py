from __future__ import annotations

import ctypes
import gc
import sys
import threading
from array import array
from collections.abc import Iterable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import treemendous
from treemendous import MutationResult, RangeSnapshot, Span, create_range_set
from treemendous.experimental.exact_batch import (
    ExactBatchRangeSet,
    MutationOpcode,
    PackedMutationResults,
)


def packed(rows: Iterable[tuple[int, int, int]]) -> array[int]:
    return array("q", (value for row in rows for value in row))


class _BigEndianInt64(ctypes.BigEndianStructure):
    _fields_ = [("value", ctypes.c_int64)]


def stable_replay(rows: list[tuple[int, int, int]], initially_available: bool = False):
    ranges = create_range_set(
        (0, 64), backend="py_boundary", initially_available=initially_available
    )
    results = []
    for opcode, start, end in rows:
        if opcode == MutationOpcode.ADD:
            results.append(ranges.add(Span(start, end)))
        else:
            results.append(
                ranges.discard(
                    Span(start, end),
                    require_covered=opcode == MutationOpcode.DISCARD_REQUIRE_COVERED,
                )
            )
    return tuple(results), ranges.snapshot()


def test_ordered_exact_results_and_canonical_snapshot() -> None:
    rows = [
        (0, 1, 5),
        (0, 3, 8),
        (0, 3, 8),  # duplicate/no-op stays represented
        (1, 4, 6),
        (2, 2, 7),  # strict rejection
        (1, 0, 10),
    ]
    manager = ExactBatchRangeSet((0, 64), initially_available=False)
    result = manager.mutate_packed(packed(rows))
    expected_results, expected_snapshot = stable_replay(rows)

    assert isinstance(result, PackedMutationResults)
    assert len(result) == len(rows)
    assert result.materialize() == expected_results
    assert all(type(item) is MutationResult for item in result.materialize())
    assert all(
        type(span) is Span for item in result.materialize() for span in item.changed
    )
    assert manager.snapshot() == expected_snapshot
    assert type(manager.snapshot()) is RangeSnapshot


def test_csr_buffers_are_owned_read_only_and_survive_owner_lifetimes() -> None:
    manager = ExactBatchRangeSet((0, 20), initially_available=False)
    result = manager.mutate_packed(packed([(0, 0, 10), (1, 3, 7), (0, 4, 6)]))
    offsets = result.changed_offsets
    spans = result.changed_spans
    lengths = result.changed_lengths
    covered = result.fully_covered
    assert offsets.format == "Q" and offsets.tolist() == [0, 1, 2, 3]
    assert spans.format == "q" and spans.ndim == 2
    assert spans.shape is not None
    assert spans.shape[1] == 2
    assert lengths.format == "q" and lengths.tolist() == [10, 4, 2]
    assert covered.format == "B" and covered.tolist() == [0, 1, 0]
    assert offsets.tolist()[-1] == spans.shape[0]
    assert all(view.readonly for view in (offsets, spans, lengths, covered))
    with pytest.raises(TypeError):
        offsets[0] = 99
    del manager, result
    gc.collect()
    assert offsets.tolist() == [0, 1, 2, 3]
    assert spans.tolist() == [[0, 10], [3, 7], [4, 6]]


def test_empty_batch_and_two_dimensional_layout() -> None:
    manager = ExactBatchRangeSet((0, 8), initially_available=False)
    empty = manager.mutate_packed(array("q"))
    assert len(empty) == 0
    assert empty.changed_offsets.tolist() == [0]
    assert empty.changed_spans.tolist() == []
    assert empty.changed_lengths.tolist() == []
    assert empty.fully_covered.tolist() == []
    storage = packed([(0, 1, 3), (1, 2, 3)])
    matrix = memoryview(storage).cast("B").cast("q", shape=(2, 3))
    assert manager.mutate_packed(matrix).materialize()[1].changed == (Span(2, 3),)


@pytest.mark.parametrize(
    "value",
    [
        [(0, 0, 1)],
        (0, 0, 1),
        b"\0" * 24,
        array("i", [0, 0, 1]),
        array("Q", [0, 0, 1]),
        (_BigEndianInt64 * 3)(),
        memoryview(array("q", range(6)))[::2],
        memoryview(array("q", range(6))).cast("B").cast("q", shape=(3, 2)),
    ],
)
def test_invalid_buffers_are_rejected(value: object) -> None:
    manager = ExactBatchRangeSet((0, 8), initially_available=False)
    before = manager.snapshot()
    with pytest.raises((TypeError, ValueError, BufferError)):
        manager.mutate_packed(value)
    assert manager.snapshot() == before


@pytest.mark.parametrize(
    "row,error",
    [
        ((9, 0, 1), ValueError),
        ((0, 1, 1), ValueError),
        ((0, -1, 1), ValueError),
        ((0, 7, 9), ValueError),
        ((0, -(1 << 63), (1 << 63) - 1), OverflowError),
    ],
)
def test_indexed_operation_errors_are_atomic(
    row: tuple[int, int, int], error: type[Exception]
) -> None:
    manager = ExactBatchRangeSet((0, 8), initially_available=False)
    before = manager.snapshot()
    with pytest.raises(error, match="operation 1"):
        manager.mutate_packed(packed([(0, 1, 2), row]))
    assert manager.snapshot() == before


@pytest.mark.parametrize("failure_index", range(5))
def test_failure_after_each_staged_prefix_is_whole_batch_atomic(
    failure_index: int,
) -> None:
    valid = [(0, index, index + 1) for index in range(5)]
    rows = valid.copy()
    rows[failure_index] = (99, 0, 1)
    manager = ExactBatchRangeSet((0, 8), initially_available=False)
    before = manager.snapshot()
    with pytest.raises(ValueError, match=rf"operation {failure_index}"):
        manager.mutate_packed(packed(rows))
    assert manager.snapshot() == before


def test_domain_int64_and_aggregate_overflow() -> None:
    with pytest.raises(OverflowError):
        ExactBatchRangeSet((-(1 << 63) - 1, 0))
    with pytest.raises(OverflowError):
        ExactBatchRangeSet((0, 1 << 63))
    with pytest.raises(OverflowError, match="measure"):
        ExactBatchRangeSet([(-(1 << 63), -1), (0, 2)])


class ReentrantExporter:
    def __init__(self, manager: ExactBatchRangeSet, *, catch: bool) -> None:
        self.manager = manager
        self.catch = catch
        self.storage = packed([(0, 1, 2)])
        self.observed: Exception | None = None

    def __buffer__(self, flags: int) -> memoryview:
        del flags
        try:
            self.manager.mutate_packed(self.storage)
        except RuntimeError as error:
            self.observed = error
            if not self.catch:
                raise
        return memoryview(self.storage)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Python-level buffer exporters require PEP 688"
)
def test_buffer_exporter_reentrancy_caught_and_uncaught() -> None:
    manager = ExactBatchRangeSet((0, 8), initially_available=False)
    caught = ReentrantExporter(manager, catch=True)
    manager.mutate_packed(caught)
    assert isinstance(caught.observed, RuntimeError)
    before = manager.snapshot()
    uncaught = ReentrantExporter(manager, catch=False)
    with pytest.raises(RuntimeError, match="reentrant"):
        manager.mutate_packed(uncaught)
    assert manager.snapshot() == before


class BlockingExporter:
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self.entered = entered
        self.release = release
        self.storage = packed([(0, 1, 2)])

    def __buffer__(self, flags: int) -> memoryview:
        del flags
        self.entered.set()
        assert self.release.wait(timeout=5)
        return memoryview(self.storage)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Python-level buffer exporters require PEP 688"
)
def test_overlapping_same_instance_rejected_and_snapshot_is_pre_or_post() -> None:
    manager = ExactBatchRangeSet((0, 8), initially_available=False)
    entered = threading.Event()
    release = threading.Event()
    thread = threading.Thread(
        target=manager.mutate_packed, args=(BlockingExporter(entered, release),)
    )
    thread.start()
    assert entered.wait(timeout=5)
    assert manager.snapshot().intervals == ()  # exporter precedes publication lock
    with pytest.raises(RuntimeError, match="overlapping"):
        manager.mutate_packed(packed([(0, 3, 4)]))
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert manager.snapshot().intervals[0].span == Span(1, 2)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Python-level buffer exporters require PEP 688"
)
def test_different_instances_can_enter_exporters_concurrently() -> None:
    left = ExactBatchRangeSet((0, 8), initially_available=False)
    right = ExactBatchRangeSet((0, 8), initially_available=False)
    barrier = threading.Barrier(2)

    class BarrierExporter:
        def __init__(self) -> None:
            self.storage = packed([(0, 1, 2)])

        def __buffer__(self, flags: int) -> memoryview:
            del flags
            barrier.wait(timeout=5)
            return memoryview(self.storage)

    threads = [
        threading.Thread(target=manager.mutate_packed, args=(BarrierExporter(),))
        for manager in (left, right)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert left.snapshot() == right.snapshot()


def test_one_large_batch_equals_ordered_sub_batches() -> None:
    rows = [(index % 3, (index * 7) % 56, (index * 7) % 56 + 8) for index in range(200)]
    whole = ExactBatchRangeSet((0, 64), initially_available=False)
    split = ExactBatchRangeSet((0, 64), initially_available=False)
    whole_results = whole.mutate_packed(packed(rows)).materialize()
    split_results = tuple(
        result
        for offset in range(0, len(rows), 13)
        for result in split.mutate_packed(
            packed(rows[offset : offset + 13])
        ).materialize()
    )
    assert split_results == whole_results
    assert split.snapshot() == whole.snapshot()


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=2),
            st.integers(min_value=0, max_value=63),
            st.integers(min_value=1, max_value=64),
        ).filter(lambda row: row[1] < row[2]),
        max_size=80,
    ),
    st.booleans(),
)
@settings(max_examples=75, deadline=None)
def test_hypothesis_differential_ordered_replay(
    rows: list[tuple[int, int, int]], initially_available: bool
) -> None:
    exact = ExactBatchRangeSet((0, 64), initially_available=initially_available)
    observed = exact.mutate_packed(packed(rows)).materialize()
    expected, final = stable_replay(rows, initially_available)
    assert observed == expected
    assert exact.snapshot() == final


def test_public_surface_is_contained() -> None:
    assert "ExactBatchRangeSet" not in treemendous.__all__
    assert not hasattr(treemendous, "ExactBatchRangeSet")
    from treemendous.backends.types import Capability
    from treemendous.protocols import RangeSetProtocol

    assert "EXACT_BATCH" not in Capability.__members__
    assert "mutate_packed" not in RangeSetProtocol.__dict__
