from __future__ import annotations

import gc
import importlib
import os
import signal
import struct
import sys
import threading
import time
from array import array
from collections.abc import Iterable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import treemendous
from treemendous import MutationResult, RangeSnapshot, Span, create_range_set
from treemendous.exact_batch import (
    BatchLimitError,
    BatchLimits,
    BatchMutation,
    ExactBatchRangeSet,
    MutationOpcode,
    PackedMutationResults,
)


def packed(rows: Iterable[tuple[int, int, int]]) -> bytes:
    values = array("q", (value for row in rows for value in row))
    assert values.itemsize == 8
    return values.tobytes()


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
        (0, 3, 8),
        (1, 4, 6),
        (2, 2, 7),
        (1, 0, 10),
    ]
    manager = ExactBatchRangeSet((0, 64), initially_available=False)
    result = manager.mutate_packed(packed(rows))
    expected_results, expected_snapshot = stable_replay(rows)

    assert isinstance(result, PackedMutationResults)
    assert result.materialize() == expected_results
    assert all(type(item) is MutationResult for item in result.materialize())
    assert all(
        type(span) is Span for item in result.materialize() for span in item.changed
    )
    assert manager.snapshot() == expected_snapshot
    assert type(manager.snapshot()) is RangeSnapshot


def test_ergonomic_batch_mutation_is_immutable_validated_and_exact() -> None:
    operation = BatchMutation(MutationOpcode.ADD, 1, 5)
    assert operation == BatchMutation(0, 1, 5)
    with pytest.raises(AttributeError):
        operation.start = 2  # type: ignore[misc]
    manager = ExactBatchRangeSet((0, 10), initially_available=False)
    results = manager.mutate([operation, BatchMutation(MutationOpcode.DISCARD, 2, 4)])
    expected = (
        MutationResult((Span(1, 5),), 4, False),
        MutationResult((Span(2, 4),), 2, True),
    )
    assert results == expected
    with pytest.raises(TypeError):
        manager.mutate([(0, 1, 2)])  # type: ignore[list-item]


@pytest.mark.parametrize("opcode", [True, "0", 3, -1])
def test_batch_mutation_rejects_invalid_opcodes(opcode: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BatchMutation(opcode, 0, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "start,end,error",
    [
        (True, 2, TypeError),
        (0, False, TypeError),
        (1, 1, ValueError),
        (2, 1, ValueError),
        (-(1 << 63) - 1, 0, OverflowError),
        (0, 1 << 63, OverflowError),
    ],
)
def test_batch_mutation_rejects_invalid_spans(
    start: object, end: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        BatchMutation(MutationOpcode.ADD, start, end)  # type: ignore[arg-type]


def test_opcode_and_native_abi_golden_values() -> None:
    expected_opcodes = (
        MutationOpcode.ADD,
        MutationOpcode.DISCARD,
        MutationOpcode.DISCARD_REQUIRE_COVERED,
    )
    assert tuple(MutationOpcode) == expected_opcodes
    assert [member.value for member in MutationOpcode] == [0, 1, 2]
    expected = struct.pack("@qqq", 2, -7, 11)
    assert len(expected) == 24
    actual = packed([(2, -7, 11)])
    assert actual == expected


def test_csr_buffers_are_owned_read_only_and_survive_owner_lifetimes() -> None:
    manager = ExactBatchRangeSet((0, 20), initially_available=False)
    result = manager.mutate_packed(packed([(0, 0, 10), (1, 3, 7), (0, 4, 6)]))
    offsets = result.changed_offsets
    spans = result.changed_spans
    lengths = result.changed_lengths
    covered = result.fully_covered
    assert offsets.format == "Q" and offsets.tolist() == [0, 1, 2, 3]
    assert spans.format == "q"
    expected_shape = (3, 2)
    assert spans.shape == expected_shape
    assert lengths.format == "q" and lengths.tolist() == [10, 4, 2]
    assert covered.format == "B" and covered.tolist() == [0, 1, 0]
    views = [offsets, spans, lengths, covered]
    assert all(view.readonly for view in views)
    with pytest.raises(TypeError):
        offsets[0] = 99
    del manager, result
    gc.collect()
    assert offsets.tolist() == [0, 1, 2, 3]
    assert spans.tolist() == [[0, 10], [3, 7], [4, 6]]


def test_empty_batch_and_exact_bytes_contract() -> None:
    manager = ExactBatchRangeSet((0, 8), initially_available=False)
    empty = manager.mutate_packed(b"")
    assert len(empty) == 0
    assert empty.changed_offsets.tolist() == [0]
    assert empty.changed_spans.tolist() == []
    assert empty.changed_lengths.tolist() == []
    assert empty.fully_covered.tolist() == []
    for invalid in (
        bytearray(24),
        memoryview(bytes(24)),
        array("q", [0, 0, 1]),
        [(0, 0, 1)],
        (0, 0, 1),
    ):
        before = manager.snapshot()
        with pytest.raises(TypeError, match="exact immutable bytes"):
            manager.mutate_packed(invalid)  # type: ignore[arg-type]
        assert manager.snapshot() == before
    with pytest.raises(ValueError, match="multiple of 24"):
        manager.mutate_packed(b"\0" * 23)


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
    rows = [(0, index, index + 1) for index in range(5)]
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


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"max_operations": True}, TypeError),
        ({"max_operations": 1.5}, TypeError),
        ({"max_operations": 0}, ValueError),
        ({"max_operations": -1}, ValueError),
        ({"max_operations": sys.maxsize + 1}, OverflowError),
    ],
)
def test_batch_limits_validate_positive_signed_size_values(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        BatchLimits(**kwargs)  # type: ignore[arg-type]


def test_operation_limit_at_minus_exact_plus() -> None:
    rows = packed([(0, 0, 1), (0, 2, 3)])
    for limit, passes in ((1, False), (2, True), (3, True)):
        manager = ExactBatchRangeSet(
            (0, 8),
            initially_available=False,
            limits=BatchLimits(max_operations=limit),
        )
        before = manager.snapshot()
        if passes:
            assert len(manager.mutate_packed(rows)) == 2
        else:
            with pytest.raises(BatchLimitError, match="max_operations"):
                manager.mutate_packed(rows)
            assert manager.snapshot() == before


def test_live_interval_limit_at_minus_exact_plus() -> None:
    with pytest.raises(BatchLimitError, match="max_live_intervals"):
        ExactBatchRangeSet([(0, 1), (2, 3)], limits=BatchLimits(max_live_intervals=1))
    row = packed([(1, 3, 5)])  # one live interval is split into exactly two
    for limit, passes in ((1, False), (2, True), (3, True)):
        manager = ExactBatchRangeSet(
            (0, 8),
            initially_available=True,
            limits=BatchLimits(max_live_intervals=limit),
        )
        before = manager.snapshot()
        if passes:
            manager.mutate_packed(row)
            assert len(manager.snapshot().intervals) == 2
        else:
            with pytest.raises(BatchLimitError, match="max_live_intervals"):
                manager.mutate_packed(row)
            assert manager.snapshot() == before


def test_changed_span_limit_at_minus_exact_plus() -> None:
    rows = packed([(0, 0, 1), (0, 2, 3)])  # exactly two cumulative deltas
    for limit, passes in ((1, False), (2, True), (3, True)):
        manager = ExactBatchRangeSet(
            (0, 8),
            initially_available=False,
            limits=BatchLimits(max_changed_spans=limit),
        )
        before = manager.snapshot()
        if passes:
            assert manager.mutate_packed(rows).changed_offsets.tolist() == [0, 1, 2]
        else:
            with pytest.raises(BatchLimitError, match="max_changed_spans"):
                manager.mutate_packed(rows)
            assert manager.snapshot() == before


def test_result_byte_accounting_at_minus_exact_plus() -> None:
    row = packed([(0, 0, 1)])
    for limit, passes in ((40, False), (41, True), (42, True)):
        manager = ExactBatchRangeSet(
            (0, 8),
            initially_available=False,
            limits=BatchLimits(max_result_bytes=limit),
        )
        before = manager.snapshot()
        if passes:
            result = manager.mutate_packed(row)
            assert (
                result.changed_offsets.nbytes
                + result.changed_spans.nbytes
                + result.changed_lengths.nbytes
                + result.fully_covered.nbytes
                == 41
            )
        else:
            with pytest.raises(BatchLimitError, match="max_result_bytes"):
                manager.mutate_packed(row)
            assert manager.snapshot() == before


def test_work_unit_limit_at_minus_exact_plus() -> None:
    rows = packed([(0, 0, 1), (1, 0, 1)])  # row costs are 1 then 2
    for limit, passes in ((2, False), (3, True), (4, True)):
        manager = ExactBatchRangeSet(
            (0, 8),
            initially_available=False,
            limits=BatchLimits(max_work_units=limit),
        )
        before = manager.snapshot()
        if passes:
            manager.mutate_packed(rows)
            assert manager.snapshot() == before
        else:
            with pytest.raises(BatchLimitError, match="max_work_units"):
                manager.mutate_packed(rows)
            assert manager.snapshot() == before


@pytest.mark.parametrize(
    "failpoint",
    [
        "operations_copy",
        "state_copy",
        "result_reserve",
        "row_staging",
        "packed_storage",
        "python_bytes",
        "wrapper_preparation",
    ],
)
def test_native_allocation_failpoints_preserve_exact_pre_state(failpoint: str) -> None:
    manager = ExactBatchRangeSet((0, 8), initially_available=False)
    before = manager.snapshot()
    manager._manager._set_failpoint(failpoint)
    with pytest.raises(MemoryError):
        manager.mutate_packed(packed([(0, 1, 2)]))
    assert manager.snapshot() == before


def test_rejected_segmented_storage_hooks_are_not_shipped() -> None:
    manager = ExactBatchRangeSet((0, 64), initially_available=True)
    assert not hasattr(manager._manager, "_storage_counters")
    for failpoint in (
        "directory_clone",
        "block_rebuild",
        "block_split",
        "directory_replacement",
        "root_creation",
    ):
        with pytest.raises(ValueError, match="unknown exact-batch failpoint"):
            manager._manager._set_failpoint(failpoint)


def test_ergonomic_materialization_failure_preserves_exact_pre_state() -> None:
    manager = ExactBatchRangeSet((0, 8), initially_available=False)
    before = manager.snapshot()
    manager._manager._set_failpoint("materialized_results")
    with pytest.raises(MemoryError):
        manager.mutate([BatchMutation(MutationOpcode.ADD, 1, 2)])
    assert manager.snapshot() == before


def test_overlapping_same_instance_is_rejected() -> None:
    manager = ExactBatchRangeSet(
        (0, 8),
        initially_available=False,
        limits=BatchLimits(max_operations=1_000_000, max_work_units=2_000_000),
    )
    rows = packed((index % 2, 0, 1) for index in range(1_000_000))
    started = threading.Event()
    finished = threading.Event()

    def mutate() -> None:
        started.set()
        manager.mutate_packed(rows)
        finished.set()

    thread = threading.Thread(target=mutate)
    thread.start()
    assert started.wait(timeout=5)
    deadline = time.monotonic() + 5
    observed_overlap = False
    while not finished.is_set() and time.monotonic() < deadline:
        with pytest.raises(RuntimeError, match="overlapping"):
            manager.mutate_packed(b"")
        observed_overlap = True
        break
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert observed_overlap


def test_snapshot_proceeds_during_long_staged_mutation() -> None:
    manager = ExactBatchRangeSet(
        (0, 8),
        initially_available=False,
        limits=BatchLimits(max_operations=1_000_000, max_work_units=2_000_000),
    )
    rows = packed((index % 2, 0, 1) for index in range(1_000_000))
    done = threading.Event()

    def mutate() -> None:
        manager.mutate_packed(rows)
        done.set()

    thread = threading.Thread(target=mutate, daemon=True)
    before = manager.snapshot()
    thread.start()
    observed_while_running = False
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not done.is_set():
        assert manager.snapshot() == before
        observed_while_running = True
        break
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert observed_while_running
    assert manager.snapshot() == before


def test_different_instances_mutate_concurrently() -> None:
    managers = [ExactBatchRangeSet((0, 8), initially_available=False) for _ in range(2)]
    barrier = threading.Barrier(2)

    def mutate(manager: ExactBatchRangeSet) -> None:
        barrier.wait(timeout=5)
        manager.mutate_packed(packed([(0, 1, 2)]))

    threads = [threading.Thread(target=mutate, args=(manager,)) for manager in managers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert managers[0].snapshot() == managers[1].snapshot()


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


_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_MALFORMED_DOMAIN = (
    (_INT64_MIN, _INT64_MIN + 32),
    (-32, 32),
    (_INT64_MAX - 32, _INT64_MAX),
)


def _multi_component_replay(rows: list[tuple[int, int, int]]):
    ranges = create_range_set(
        _MALFORMED_DOMAIN,
        backend="py_boundary",
        initially_available=False,
    )
    results = []
    for opcode, start, end in rows:
        span = Span(start, end)
        if opcode == MutationOpcode.ADD:
            results.append(ranges.add(span))
        else:
            results.append(
                ranges.discard(
                    span,
                    require_covered=(opcode == MutationOpcode.DISCARD_REQUIRE_COVERED),
                )
            )
    return tuple(results), ranges.snapshot()


@given(st.binary(max_size=24 * 12))
@settings(max_examples=100, deadline=None)
def test_malformed_immutable_bytes_never_publish_partial_state(blob: bytes) -> None:
    manager = ExactBatchRangeSet(_MALFORMED_DOMAIN, initially_available=False)
    before = manager.snapshot()
    try:
        observed = manager.mutate_packed(blob).materialize()
    except (ValueError, OverflowError):
        assert manager.snapshot() == before
        return

    rows = [tuple(row) for row in struct.iter_unpack("@qqq", blob)]
    expected, final = _multi_component_replay(rows)
    assert observed == expected
    assert manager.snapshot() == final


_VALID_EXTREME_ROWS = (
    (0, _INT64_MIN, _INT64_MIN + 1),
    (1, _INT64_MIN, _INT64_MIN + 1),
    (0, -32, 32),
    (2, -16, 16),
    (1, -32, 0),
    (0, _INT64_MAX - 1, _INT64_MAX),
    (2, _INT64_MAX - 1, _INT64_MAX),
)


@given(st.lists(st.sampled_from(_VALID_EXTREME_ROWS), max_size=40))
@settings(max_examples=75, deadline=None)
def test_extreme_int64_multi_component_bytes_match_scalar_semantics(
    rows: list[tuple[int, int, int]],
) -> None:
    manager = ExactBatchRangeSet(_MALFORMED_DOMAIN, initially_available=False)
    observed = manager.mutate_packed(packed(rows)).materialize()
    expected, final = _multi_component_replay(rows)
    assert observed == expected
    assert manager.snapshot() == final


@given(
    st.lists(st.sampled_from(_VALID_EXTREME_ROWS), min_size=1, max_size=30),
    st.one_of(
        st.tuples(
            st.sampled_from((-1, 3, 99, _INT64_MIN, _INT64_MAX)),
            st.just(-1),
            st.just(1),
        ),
        st.sampled_from(
            (
                (0, _INT64_MIN, _INT64_MAX),
                (1, -40, -20),
                (2, 0, 0),
            )
        ),
    ),
)
@settings(max_examples=75, deadline=None)
def test_late_random_opcode_or_span_failure_is_atomic(
    prefix: list[tuple[int, int, int]], invalid: tuple[int, int, int]
) -> None:
    manager = ExactBatchRangeSet(_MALFORMED_DOMAIN, initially_available=False)
    before = manager.snapshot()
    with pytest.raises((ValueError, OverflowError), match=rf"operation {len(prefix)}"):
        manager.mutate_packed(packed([*prefix, invalid]))
    assert manager.snapshot() == before


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires interval timers")
def test_signal_interrupt_rolls_back_staged_mutation() -> None:
    manager = ExactBatchRangeSet(
        (0, 8),
        initially_available=False,
        limits=BatchLimits(max_operations=1_000_000, max_work_units=2_000_000),
    )
    rows = packed((index % 2, 0, 1) for index in range(1_000_000))
    before = manager.snapshot()
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda _signum, _frame: None)
    signal.setitimer(signal.ITIMER_REAL, 0.001)
    try:
        with pytest.raises(KeyboardInterrupt):
            # SIGINT gives Python's normal KeyboardInterrupt while native rows run.
            signal.signal(
                signal.SIGALRM,
                lambda _signum, _frame: os.kill(os.getpid(), signal.SIGINT),
            )
            manager.mutate_packed(rows)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    assert manager.snapshot() == before


@pytest.mark.skipif(sys.version_info < (3, 12), reason="focused on Python 3.12+")
def test_python_312_subinterpreter_import_and_use() -> None:
    interpreters = pytest.importorskip("_xxsubinterpreters")
    interpreter = interpreters.create()
    try:
        interpreters.run_string(
            interpreter,
            "from treemendous.exact_batch import (BatchMutation, ExactBatchRangeSet, MutationOpcode); "
            "r=ExactBatchRangeSet((0,8), initially_available=False); "
            "assert r.mutate([BatchMutation(MutationOpcode.ADD,1,2)])[0].changed_length == 1",
        )
    finally:
        interpreters.destroy(interpreter)


def test_stable_module_identity_experimental_absence_and_containment() -> None:
    assert ExactBatchRangeSet.__module__ == "treemendous.exact_batch"
    assert PackedMutationResults.__module__ == "treemendous.exact_batch"
    assert BatchLimitError.__module__ == "treemendous.exact_batch"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("treemendous.experimental")
    assert "ExactBatchRangeSet" not in treemendous.__all__
    assert not hasattr(treemendous, "ExactBatchRangeSet")
    from treemendous.backends.types import Capability
    from treemendous.protocols import RangeSetProtocol

    assert "EXACT_BATCH" not in Capability.__members__
    assert "mutate_packed" not in RangeSetProtocol.__dict__
