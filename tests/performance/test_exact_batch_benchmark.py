from __future__ import annotations

from typing import Any

import pytest

from tests.performance import exact_batch_benchmark as benchmark
from treemendous import BackendUnavailableError, Span
from treemendous.exact_batch import MutationOpcode

pytestmark = pytest.mark.benchmark


def test_per_size_traces_are_restorative_without_state_contraction() -> None:
    for batch_size in benchmark.BATCH_SIZES:
        trace = benchmark.trace_for_size(batch_size)
        assert len(trace) == batch_size
        expected = benchmark._validate(trace, benchmark._packed(trace))
        exact = benchmark._new_exact()
        scalar = benchmark._new_scalar()
        for _ in range(5):
            exact_results = exact.mutate_packed(benchmark._packed(trace)).materialize()
            scalar_results = benchmark._scalar_results(scalar, trace)
            assert exact_results == scalar_results
            assert exact.snapshot() == expected
            assert scalar.snapshot() == expected
            assert len(exact.snapshot().intervals) == benchmark.INITIAL_INTERVAL_COUNT


def test_timed_instance_result_corruption_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_new_exact = benchmark._new_exact
    construction_count = 0

    class CorruptResult:
        def materialize(self) -> tuple[()]:
            return ()

    class CorruptTimedManager:
        def __init__(self, manager: Any) -> None:
            self._manager = manager

        def mutate_packed(self, operations: bytes) -> CorruptResult:
            self._manager.mutate_packed(operations)
            return CorruptResult()

        def snapshot(self) -> Any:
            return self._manager.snapshot()

    def new_exact() -> Any:
        nonlocal construction_count
        construction_count += 1
        manager = real_new_exact()
        if construction_count == 2:
            return CorruptTimedManager(manager)
        return manager

    monkeypatch.setattr(benchmark, "_new_exact", new_exact)
    with pytest.raises(AssertionError, match="timed packed per-row results"):
        benchmark.run_benchmark(samples=20, target_operations=1)


def test_primary_trace_contains_required_real_and_nonmutating_cases() -> None:
    trace4 = benchmark.trace_for_size(4)
    exact4 = benchmark._new_exact()
    results4 = exact4.mutate_packed(benchmark._packed(trace4)).materialize()
    expected_partial = (Span(2, 6),)
    expected_empty: tuple[Span, ...] = ()
    assert results4[0].changed == expected_partial
    assert results4[1].changed == expected_partial
    assert trace4[2][0] == MutationOpcode.DISCARD_REQUIRE_COVERED
    assert not results4[2].fully_covered
    assert results4[2].changed == expected_empty
    assert results4[3].changed == expected_empty

    trace8 = benchmark.trace_for_size(8)
    exact8 = benchmark._new_exact()
    results8 = exact8.mutate_packed(benchmark._packed(trace8)).materialize()
    expected_wide = (Span(8, 16), Span(24, 32), Span(40, 48))
    expected_restores = (
        (Span(8, 16),),
        (Span(24, 32),),
        (Span(40, 48),),
    )
    assert results8[4].changed == expected_wide
    assert tuple(result.changed for result in results8[5:8]) == expected_restores

    trace16 = benchmark.trace_for_size(16)
    exact16 = benchmark._new_exact()
    results16 = exact16.mutate_packed(benchmark._packed(trace16)).materialize()
    expected_overlap = (Span(132, 136),)
    assert results16[8].changed == expected_overlap  # partial overlap
    assert trace16[10] == trace16[11]  # duplicate no-ops
    assert results16[10].changed == results16[11].changed == expected_empty
    assert trace16[12] == trace16[13]  # duplicate strict rejections
    assert results16[12].changed == results16[13].changed == expected_empty


def test_batch_two_is_a_real_mutation_restore_pair_and_one_is_diagnostic() -> None:
    trace2 = benchmark.trace_for_size(2)
    exact = benchmark._new_exact()
    results = exact.mutate_packed(benchmark._packed(trace2)).materialize()
    assert results[0].changed_length == results[1].changed_length == 4
    assert exact.snapshot() == benchmark._new_exact().snapshot()
    expected_diagnostic = ((MutationOpcode.ADD, Span(0, 8)),)
    assert benchmark.trace_for_size(1) == expected_diagnostic


def test_cpp_boundary_is_the_only_scalar_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str | None] = []

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        del args
        observed.append(kwargs.get("backend"))
        raise BackendUnavailableError("deliberately unavailable")

    monkeypatch.setattr(benchmark, "create_range_set", unavailable)
    with pytest.raises(BackendUnavailableError, match="deliberately unavailable"):
        benchmark.run_benchmark(samples=20, target_operations=1)
    assert observed == ["cpp_boundary"]


def test_packed_rows_use_exact_native_int64_layout() -> None:
    packed = benchmark._packed(benchmark.trace_for_size(4))
    assert type(packed) is bytes
    assert len(packed) == 12 * 8
