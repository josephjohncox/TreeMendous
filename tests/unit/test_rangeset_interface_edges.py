from __future__ import annotations

from typing import Any

import pytest

from treemendous.backends.adapters import BackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import (
    IntervalResult,
    ManagedDomainRequiredError,
    MutationResult,
    Span,
)
from treemendous.policies import OrderedPayloadPolicy, UniformPayloadPolicy
from treemendous.rangeset import RangeSet


def _ranges(**options: Any) -> RangeSet:
    return RangeSet(BackendAdapter(IntervalManager()), **options)


class _DeltaIntervalManager:
    _treemendous_authoritative_geometry = True

    def __init__(self) -> None:
        self.free: set[int] = set()
        self.interval_reads = 0
        self.delta_releases = 0
        self.delta_reserves = 0
        self.fit_queries = 0
        self.overlap_queries = 0

    @staticmethod
    def _intervals(points: set[int]) -> list[tuple[int, int]]:
        if not points:
            return []
        ordered = sorted(points)
        result: list[tuple[int, int]] = []
        start = previous = ordered[0]
        for point in ordered[1:]:
            if point != previous + 1:
                result.append((start, previous + 1))
                start = point
            previous = point
        result.append((start, previous + 1))
        return result

    def release_interval(self, start: int, end: int) -> None:
        self.free.update(range(start, end))

    def reserve_interval(self, start: int, end: int) -> None:
        self.free.difference_update(range(start, end))

    def release_with_delta(self, start: int, end: int) -> MutationResult:
        self.delta_releases += 1
        target = set(range(start, end))
        changed = target - self.free
        changed_spans = tuple(Span(*item) for item in self._intervals(changed))
        result = MutationResult(changed_spans, len(changed), not changed)
        self.free.update(target)
        return result

    def reserve_with_delta(
        self, start: int, end: int, require_covered: bool
    ) -> MutationResult:
        self.delta_reserves += 1
        target = set(range(start, end))
        covered = target <= self.free
        if require_covered and not covered:
            return MutationResult((), 0, False)
        changed = target & self.free
        changed_spans = tuple(Span(*item) for item in self._intervals(changed))
        result = MutationResult(changed_spans, len(changed), covered)
        self.free.difference_update(target)
        return result

    def find_interval(self, start: int, length: int) -> tuple[int, int] | None:
        self.fit_queries += 1
        for interval_start, interval_end in self._intervals(self.free):
            allocation_start = max(start, interval_start)
            if allocation_start + length <= interval_end:
                return allocation_start, allocation_start + length
        return None

    def find_overlapping_intervals(self, start: int, end: int) -> list[tuple[int, int]]:
        self.overlap_queries += 1
        return [
            (interval_start, interval_end)
            for interval_start, interval_end in self._intervals(self.free)
            if interval_start < end and start < interval_end
        ]

    def get_intervals(self) -> list[tuple[int, int]]:
        self.interval_reads += 1
        return self._intervals(self.free)


class _CountingIntervalManager:
    def __init__(self) -> None:
        self._implementation = IntervalManager()
        self.interval_reads = 0

    def release_interval(self, start: int, end: int) -> None:
        self._implementation.release_interval(start, end)

    def reserve_interval(self, start: int, end: int) -> None:
        self._implementation.reserve_interval(start, end)

    def get_intervals(self) -> list[IntervalResult]:
        self.interval_reads += 1
        return self._implementation.get_intervals()


def test_authoritative_delta_backend_handles_geometry_only_hot_paths() -> None:
    implementation = _DeltaIntervalManager()
    ranges = RangeSet(BackendAdapter(implementation), domain=(0, 10))
    initial_reads = implementation.interval_reads

    discarded = ranges.discard(Span(2, 4), require_covered=True)
    expected_change = (Span(2, 4),)
    assert discarded.changed == expected_change
    assert discarded.changed_length == 2
    assert discarded.fully_covered
    added = ranges.add(Span(2, 4))
    assert added.changed == expected_change
    assert not added.fully_covered
    assert ranges.first_fit(3, not_before=1) == IntervalResult(1, 4)
    expected_overlap = (IntervalResult(0, 10),)
    assert ranges.overlaps(Span(3, 5)) == expected_overlap

    assert implementation.delta_reserves == 1
    assert implementation.delta_releases == 1
    assert implementation.fit_queries == 1
    assert implementation.overlap_queries == 1
    assert implementation.interval_reads == initial_reads


def test_authoritative_backend_materializes_only_after_a_changed_mutation() -> None:
    implementation = _DeltaIntervalManager()
    ranges = RangeSet(BackendAdapter(implementation), domain=(0, 10))
    initial_reads = implementation.interval_reads

    ranges.discard(Span(2, 4), require_covered=True)
    assert implementation.interval_reads == initial_reads
    expected = (IntervalResult(0, 2), IntervalResult(4, 10))
    assert ranges.intervals() == expected
    assert ranges.intervals() == expected
    assert implementation.interval_reads == initial_reads

    no_op = ranges.discard(Span(2, 4))
    assert not no_op.changed
    assert ranges.intervals() == expected
    assert implementation.interval_reads == initial_reads

    ranges.add(Span(2, 4))
    complete = (IntervalResult(0, 10),)
    assert ranges.intervals() == complete
    assert implementation.interval_reads == initial_reads

    ranges.discard(Span(0, 1), require_covered=True)
    ranges.discard(Span(2, 3), require_covered=True)
    expected_fragments = (IntervalResult(1, 2), IntervalResult(3, 10))
    assert ranges.intervals() == expected_fragments
    assert implementation.interval_reads == initial_reads + 1


def test_authoritative_result_construction_rejects_reentrant_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranges = _ranges(domain=(0, 10))
    original_init = MutationResult.__init__
    attempted = False

    def reenter_once(self, *args: Any, **kwargs: Any) -> None:
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(RuntimeError, match="reentrant mutation"):
                ranges.discard(Span(2, 4), require_covered=True)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(MutationResult, "__init__", reenter_once)
    result = ranges.discard(Span(2, 4), require_covered=True)
    expected_changed = (Span(2, 4),)
    expected_intervals = (IntervalResult(0, 2), IntervalResult(4, 10))
    assert result.changed == expected_changed
    assert ranges.snapshot().total_free == 8
    assert ranges.intervals() == expected_intervals


def test_delegated_domain_validation_preserves_public_error_ordering() -> None:
    ranges = _ranges(domain=(0, 4), initially_available=False)
    ranges._backend_validates_domain = True
    with pytest.raises(ValueError, match="managed domain"):
        ranges.add(Span(5, 6), payload="unexpected")
    assert not ranges.intervals()


def test_mutations_and_queries_do_not_reenumerate_unchanged_backend_state() -> None:
    implementation = _CountingIntervalManager()
    ranges = RangeSet(
        BackendAdapter(implementation),
        domain=(0, 128),
        initially_available=False,
    )
    initial_reads = implementation.interval_reads
    spans = tuple(Span(i * 2, i * 2 + 1) for i in range(64))

    for span in spans:
        ranges.add(span)
    for span in spans:
        ranges.discard(span, require_covered=True)
        ranges.add(span)
    assert ranges.first_fit(1, not_before=0) is not None
    assert ranges.overlaps(Span(0, 3))
    assert ranges.snapshot().total_free == 64
    assert ranges.stats().total_free == 64

    assert implementation.interval_reads == initial_reads


def test_geometry_aggregates_track_equal_maxima_merges_and_splits() -> None:
    ranges = _ranges(domain=(0, 20), initially_available=False)
    for span in (Span(0, 4), Span(6, 10), Span(12, 16)):
        ranges.add(span)

    assert ranges.stats().total_free == 12
    assert ranges.stats().largest_chunk == 4
    ranges.discard(Span(1, 3), require_covered=True)
    ranges.discard(Span(6, 10), require_covered=True)
    assert ranges.stats().largest_chunk == 4
    ranges.discard(Span(12, 16), require_covered=True)
    assert ranges.stats().largest_chunk == 1

    ranges.add(Span(1, 15))
    assert ranges.snapshot().total_free == 15
    assert ranges.stats().largest_chunk == 15
    assert ranges.first_fit(16, not_before=0) is None


def test_payload_operations_require_an_explicit_policy() -> None:
    ranges = _ranges(initially_available=False)
    before = ranges.snapshot()

    with pytest.raises(ValueError, match="explicit payload policy"):
        ranges.add(Span(0, 2), "cpu")
    with pytest.raises(ValueError, match="explicit payload policy"):
        ranges.first_fit(1, not_before=0, payload_predicate=lambda data: True)
    assert ranges.snapshot() == before


def test_initial_payload_state_uses_each_policy_identity() -> None:
    uniform_policy = UniformPayloadPolicy[str]()
    uniform = _ranges(
        domain=(0, 4),
        payload_policy=uniform_policy,
    )
    assert uniform.domain is not None
    assert list(uniform.domain.bounds) == [0, 4]
    assert uniform.payload_policy is uniform_policy
    assert uniform.intervals()[0].data is None

    ordered_policy = OrderedPayloadPolicy[tuple[str, ...]](
        lambda left, right: left + right,
        tuple(),
        event_key_fn=lambda value: value,
    )
    ordered = _ranges(
        domain=(0, 4),
        payload_policy=ordered_policy,
    )
    assert ordered.payload_policy is ordered_policy
    assert ordered.intervals()[0].data == tuple()


def test_rangeset_edge_paths_remain_failure_atomic_and_observable() -> None:
    ranges = _ranges(
        domain=(0, 10),
        payload_policy=UniformPayloadPolicy[str](),
        initially_available=False,
    )
    ranges.add(Span(1, 5), "A")
    before = ranges.snapshot()

    with pytest.raises(ValueError, match="contained in the managed domain"):
        ranges.add(Span(9, 12), "A")
    with pytest.raises(ValueError, match="not_after"):
        ranges.first_fit(1, not_before=4, not_after=4)
    assert ranges.snapshot() == before

    mutation = ranges.discard(Span(0, 2), require_covered=True)
    assert not mutation.changed
    assert mutation.changed_length == 0
    assert not mutation.fully_covered
    assert ranges.allocate(20, not_before=0) is None

    overlaps = ranges.overlaps(Span(3, 7))
    assert len(overlaps) == 1
    assert overlaps[0].span == Span(1, 5)
    assert ranges.intervals()[0].data == "A"
    assert ranges.snapshot().total_free == 4
    assert ranges.stats().total_free == 4


def test_stats_require_an_explicit_domain() -> None:
    ranges = _ranges(initially_available=False)
    with pytest.raises(ManagedDomainRequiredError, match="explicit managed domain"):
        ranges.stats()


def test_rangeset_has_no_raw_or_legacy_escape_hatches() -> None:
    ranges = _ranges(initially_available=False)
    for name in (
        "release_interval",
        "reserve_interval",
        "find_interval",
        "get_intervals",
        "get_raw_implementation",
        "require",
    ):
        assert not hasattr(ranges, name)
