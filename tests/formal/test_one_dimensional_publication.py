"""Executable checks for authoritative immutable-publication state transitions."""

from __future__ import annotations

from tests.formal.one_dimensional_model import ModeledMutation, release, reserve, runs
from treemendous.backends.adapters import BackendAdapter
from treemendous.domain import IntervalResult, MutationResult, Span
from treemendous.rangeset import RangeSet


class _CountingAuthoritativeManager:
    _treemendous_authoritative_geometry = True

    def __init__(self) -> None:
        self.free: frozenset[int] = frozenset()
        self.interval_reads = 0

    def set_managed_domain(self, spans: object) -> None:
        del spans

    @staticmethod
    def _result(modeled: ModeledMutation) -> MutationResult:
        return MutationResult(
            tuple(Span(start, end) for start, end in modeled.changed),
            modeled.changed_length,
            modeled.fully_covered,
        )

    def release_interval(self, start: int, end: int) -> None:
        self.free = release(self.free, (start, end)).after

    def reserve_interval(self, start: int, end: int) -> None:
        self.free = reserve(self.free, (start, end)).after

    def release_with_delta(self, start: int, end: int) -> MutationResult:
        modeled = release(self.free, (start, end))
        self.free = modeled.after
        return self._result(modeled)

    def reserve_with_delta(
        self, start: int, end: int, require_covered: bool
    ) -> MutationResult:
        modeled = reserve(self.free, (start, end), require_covered=require_covered)
        self.free = modeled.after
        return self._result(modeled)

    def find_interval(self, point: int, length: int) -> tuple[int, int] | None:
        for start, end in runs(self.free):
            allocation_start = max(point, start)
            if allocation_start + length <= end:
                return allocation_start, allocation_start + length
        return None

    def find_overlapping_intervals(self, start: int, end: int) -> list[tuple[int, int]]:
        return [
            (current_start, current_end)
            for current_start, current_end in runs(self.free)
            if current_start < end and start < current_end
        ]

    def get_intervals(self) -> list[tuple[int, int]]:
        self.interval_reads += 1
        return list(runs(self.free))


def _ranges() -> tuple[RangeSet, _CountingAuthoritativeManager]:
    raw = _CountingAuthoritativeManager()
    ranges = RangeSet(
        BackendAdapter(raw),
        domain=(0, 10),
        initially_available=False,
    )
    return ranges, raw


def test_valid_noop_patchable_dirty_and_rematerialized_transitions() -> None:
    ranges, raw = _ranges()
    assert raw.interval_reads == 1

    valid = ranges.intervals()
    assert ranges.intervals() is valid

    # V -- no-op --> V; tuple identity and enumeration count are unchanged.
    no_op = ranges.discard(Span(2, 4))
    assert not no_op.changed
    assert ranges.intervals() is valid
    assert raw.interval_reads == 1

    # V -- effective --> P+ -- intervals() --> V. Publication copies tuple
    # entries but does not enumerate the backend.
    ranges.add(Span(2, 4))
    patched = ranges.intervals()
    expected_patched = (IntervalResult(2, 4),)
    assert patched == expected_patched
    assert patched is ranges.intervals()
    assert patched is not valid
    assert raw.interval_reads == 1

    # Two unobserved effective mutations discard the single-patch recipe:
    # V -> P- -> D, then D -- intervals() --> V via one backend read.
    ranges.discard(Span(2, 3))
    ranges.add(Span(6, 8))
    assert raw.interval_reads == 1
    rematerialized = ranges.intervals()
    expected_rematerialized = (IntervalResult(3, 4), IntervalResult(6, 8))
    assert rematerialized == expected_rematerialized
    assert raw.interval_reads == 2
    assert ranges.intervals() is rematerialized
    assert raw.interval_reads == 2


def test_snapshot_reuses_publication_but_validates_the_whole_tuple() -> None:
    ranges, raw = _ranges()
    ranges.add(Span(0, 2))
    ranges.add(Span(4, 6))
    assert raw.interval_reads == 1

    snapshot = ranges.snapshot()
    expected = (IntervalResult(0, 2), IntervalResult(4, 6))
    assert snapshot.intervals == expected
    assert snapshot.total_free == 4
    assert raw.interval_reads == 2
    assert ranges.snapshot().intervals is snapshot.intervals
    assert raw.interval_reads == 2
