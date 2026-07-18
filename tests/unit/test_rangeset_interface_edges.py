from __future__ import annotations

from typing import Any

import pytest

from treemendous.backends.adapters import BackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import ManagedDomainRequiredError, Span
from treemendous.policies import OrderedPayloadPolicy, UniformPayloadPolicy
from treemendous.rangeset import RangeSet


def _ranges(**options: Any) -> RangeSet:
    return RangeSet(BackendAdapter(IntervalManager()), **options)


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
