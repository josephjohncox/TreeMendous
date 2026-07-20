"""Security regressions for arbitrary payload code re-entering RangeSet."""

from __future__ import annotations

from threading import Thread
from typing import Any

import pytest

from treemendous.backends.adapters import BackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import Span
from treemendous.policies import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    UniformPayloadPolicy,
)
from treemendous.rangeset import RangeSet


def _ranges(policy: Any) -> RangeSet:
    return RangeSet(
        BackendAdapter(IntervalManager()),
        initially_available=False,
        payload_policy=policy,
    )


def test_combine_callback_cannot_mutate_rangeset() -> None:
    ranges: RangeSet

    def combine(left: list[str], right: list[str]) -> list[str]:
        ranges.discard(Span(0, 10))
        return left + right

    ranges = _ranges(JoinPayloadPolicy(combine, []))
    ranges.add(Span(0, 10), ["A"])
    before = ranges.snapshot()

    with pytest.raises(RuntimeError, match="payload processing"):
        ranges.add(Span(5, 15), ["B"])

    assert ranges.snapshot() == before


def test_cross_thread_callback_mutation_is_rejected_without_deadlock() -> None:
    ranges: RangeSet
    errors: list[BaseException] = []

    def combine(left: list[str], right: list[str]) -> list[str]:
        def mutate() -> None:
            try:
                ranges.discard(Span(0, 1))
            except BaseException as exc:  # capture worker evidence
                errors.append(exc)

        worker = Thread(target=mutate)
        worker.start()
        worker.join(timeout=1)
        if worker.is_alive():
            raise RuntimeError("cross-thread payload mutation deadlocked")
        return left + right

    ranges = _ranges(JoinPayloadPolicy(combine, []))
    ranges.add(Span(0, 10), ["A"])
    ranges.add(Span(5, 15), ["B"])

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "payload processing" in str(errors[0])
    observed_geometry = [(item.start, item.end) for item in ranges.intervals()]
    expected_geometry = [(0, 5), (5, 10), (10, 15)]
    assert observed_geometry == expected_geometry


def test_restrict_and_event_key_callbacks_cannot_mutate_rangeset() -> None:
    restricted: RangeSet

    def restrict(data: list[str], source: Span, target: Span) -> list[str]:
        restricted.add(Span(20, 21), ["nested"])
        return data

    restricted = _ranges(
        JoinPayloadPolicy(lambda left, right: left + right, [], restrict)
    )
    with pytest.raises(RuntimeError, match="payload processing"):
        restricted.add(Span(0, 10), ["A"])
    assert not restricted.intervals()

    ordered: RangeSet

    def event_key(data: tuple[str, ...]) -> tuple[str, ...]:
        ordered.add(Span(20, 21), ("nested",))
        return data

    policy = OrderedPayloadPolicy(
        lambda left, right: left + right,
        (),
        event_key_fn=event_key,
    )
    ordered = _ranges(policy)
    with pytest.raises(RuntimeError, match="payload processing"):
        ordered.add(Span(0, 10), ("A",))
    assert not ordered.intervals()


@pytest.mark.parametrize("method", ["first_fit", "allocate"])
def test_payload_predicate_cannot_mutate_rangeset(method: str) -> None:
    ranges = _ranges(JoinPayloadPolicy(lambda left, right: left + right, []))
    ranges.add(Span(0, 10), ["A"])
    before = ranges.snapshot()

    def predicate(data: list[str]) -> bool:
        ranges.discard(Span(0, 10))
        return bool(data)

    with pytest.raises(RuntimeError, match="payload processing"):
        getattr(ranges, method)(5, not_before=0, payload_predicate=predicate)

    assert ranges.snapshot() == before


def test_no_backend_read_occurs_after_payload_mutation_commit_point() -> None:
    class ReadFaultAfterMutation:
        def __init__(self) -> None:
            self.delegate = IntervalManager()
            self.fail_reads = False

        def release_interval(self, start: int, end: int) -> None:
            self.delegate.release_interval(start, end)
            self.fail_reads = True

        def reserve_interval(self, start: int, end: int) -> None:
            self.delegate.reserve_interval(start, end)
            self.fail_reads = True

        def get_intervals(self):
            if self.fail_reads:
                raise RuntimeError("post-mutation read")
            return self.delegate.get_intervals()

    raw = ReadFaultAfterMutation()
    adapter = BackendAdapter(raw)
    ranges = RangeSet(
        adapter,
        initially_available=False,
        payload_policy=JoinPayloadPolicy(lambda left, right: left + right, []),
    )

    added = ranges.add(Span(0, 10), ["A"])
    assert added.changed_length == 10
    raw.fail_reads = False
    added_geometry = [(item.start, item.end) for item in adapter.intervals()]
    expected_added_geometry = [(0, 10)]
    assert added_geometry == expected_added_geometry

    removed = ranges.discard(Span(2, 8))
    assert removed.changed_length == 6
    raw.fail_reads = False
    remaining_geometry = [(item.start, item.end) for item in adapter.intervals()]
    expected_geometry = [(0, 2), (8, 10)]
    assert remaining_geometry == expected_geometry


def test_constructor_payload_failure_precedes_backend_mutation() -> None:
    raw = IntervalManager()

    def fail_key(data: tuple[str, ...]) -> tuple[str, ...]:
        raise RuntimeError("key failed")

    with pytest.raises(RuntimeError, match="key failed"):
        RangeSet(
            BackendAdapter(raw),
            domain=((0, 2), (4, 6)),
            payload_policy=OrderedPayloadPolicy(
                lambda left, right: left + right,
                (),
                event_key_fn=fail_key,
            ),
        )

    assert not raw.get_intervals()


def test_deepcopy_equality_repr_and_order_hooks_cannot_mutate_rangeset() -> None:
    deepcopy_ranges = _ranges(UniformPayloadPolicy())

    class CopyHook:
        def __deepcopy__(self, memo: dict[int, Any]) -> CopyHook:
            deepcopy_ranges.add(Span(20, 21), "nested")
            return self

    with pytest.raises(RuntimeError, match="payload processing"):
        deepcopy_ranges.add(Span(0, 10), CopyHook())
    assert not deepcopy_ranges.intervals()

    equality_ranges = _ranges(UniformPayloadPolicy())

    class EqualityHook:
        def __deepcopy__(self, memo: dict[int, Any]) -> EqualityHook:
            return self

        def __eq__(self, other: object) -> bool:
            equality_ranges.discard(Span(0, 1))
            return True

    equality_ranges.add(Span(0, 1), EqualityHook())
    before_geometry = tuple(
        (item.start, item.end) for item in equality_ranges.intervals()
    )
    with pytest.raises(RuntimeError, match="payload processing"):
        equality_ranges.add(Span(1, 2), EqualityHook())
    observed_geometry = tuple(
        (item.start, item.end) for item in equality_ranges.intervals()
    )
    assert observed_geometry == before_geometry

    repr_ranges: RangeSet

    class ReprHook:
        def __deepcopy__(self, memo: dict[int, Any]) -> ReprHook:
            return self

        def __repr__(self) -> str:
            repr_ranges.add(Span(20, 21), self)
            return "payload"

    repr_ranges = _ranges(OrderedPayloadPolicy(lambda left, right: left, ReprHook()))
    with pytest.raises(RuntimeError, match="payload processing"):
        repr_ranges.add(Span(0, 1), ReprHook())
    assert not repr_ranges.intervals()

    ordering_ranges: RangeSet

    class OrderingKey:
        def __deepcopy__(self, memo: dict[int, Any]) -> OrderingKey:
            return self

        def __lt__(self, other: object) -> bool:
            ordering_ranges.discard(Span(0, 1))
            return False

    ordering_ranges = _ranges(
        OrderedPayloadPolicy(
            lambda left, right: left + right,
            (),
            event_key_fn=lambda data: OrderingKey(),
        )
    )
    ordering_ranges.add(Span(0, 2), ("A",))
    before = ordering_ranges.snapshot()
    with pytest.raises(RuntimeError, match="payload processing"):
        ordering_ranges.add(Span(0, 2), ("B",))
    assert ordering_ranges.snapshot() == before
