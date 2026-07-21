"""Finite refinement of every stable backend against the point-set model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from tests.formal.one_dimensional_model import (
    all_point_sets,
    release,
    reserve,
    runs,
    valid_spans,
)
from treemendous import BackendRegistry, MutationResult, RangeSet, Span
from treemendous.backends.types import Available, Maturity

EXTENT = 5
REGISTRY = BackendRegistry.discover()
STABLE_BACKENDS = tuple(
    spec.id
    for spec in REGISTRY.specs
    if spec.maturity is Maturity.STABLE
    and isinstance(REGISTRY.states[spec.id], Available)
)


def _seed(backend_id: str, free: frozenset[int]) -> RangeSet:
    ranges = REGISTRY.create(
        (0, EXTENT),
        backend=backend_id,
        initially_available=False,
    )
    for start, end in runs(free):
        ranges.add(Span(start, end))
    return ranges


def _pairs(items: Iterable[Any]) -> tuple[tuple[int, int], ...]:
    return tuple((item.start, item.end) for item in items)


def _raw_points(ranges: RangeSet) -> frozenset[int]:
    """Read backend geometry directly, bypassing RangeSet's fallback cache."""
    return frozenset(
        point
        for item in ranges._adapter.intervals()
        for point in range(item.start, item.end)
    )


def _assert_result(
    observed: MutationResult,
    expected_changed: tuple[tuple[int, int], ...],
    expected_length: int,
    expected_covered: bool,
) -> None:
    assert type(observed) is MutationResult
    assert type(observed.changed) is tuple
    assert all(type(span) is Span for span in observed.changed)
    assert _pairs(observed.changed) == expected_changed
    assert observed.changed_length == expected_length
    assert observed.fully_covered is expected_covered


@pytest.mark.parametrize("backend_id", STABLE_BACKENDS)
def test_every_small_state_and_mutation_refines_the_finite_model(
    backend_id: str,
) -> None:
    for free in all_point_sets(EXTENT):
        for target in valid_spans(EXTENT):
            released_model = release(free, target)
            released = _seed(backend_id, free)
            released_result = released.add(Span(*target))
            _assert_result(
                released_result,
                released_model.changed,
                released_model.changed_length,
                released_model.fully_covered,
            )
            assert _raw_points(released) == released_model.after
            assert _pairs(released.intervals()) == runs(released_model.after)

            reserved_model = reserve(free, target)
            reserved = _seed(backend_id, free)
            reserved_result = reserved.discard(Span(*target))
            _assert_result(
                reserved_result,
                reserved_model.changed,
                reserved_model.changed_length,
                reserved_model.fully_covered,
            )
            assert _raw_points(reserved) == reserved_model.after
            assert _pairs(reserved.intervals()) == runs(reserved_model.after)

            rejected_model = reserve(free, target, require_covered=True)
            rejected = _seed(backend_id, free)
            rejected_result = rejected.discard(Span(*target), require_covered=True)
            _assert_result(
                rejected_result,
                rejected_model.changed,
                rejected_model.changed_length,
                rejected_model.fully_covered,
            )
            assert _raw_points(rejected) == rejected_model.after
            assert _pairs(rejected.intervals()) == runs(rejected_model.after)


@pytest.mark.parametrize("backend_id", STABLE_BACKENDS)
def test_public_output_is_canonical_across_adjacency_and_disconnected_domains(
    backend_id: str,
) -> None:
    ranges = REGISTRY.create(
        ((0, 2), (3, 5)),
        backend=backend_id,
        initially_available=False,
    )
    ranges.add(Span(0, 1))
    ranges.add(Span(1, 2))
    expected_intervals = ((0, 2),)
    assert _pairs(ranges.intervals()) == expected_intervals

    before = ranges.snapshot()
    with pytest.raises(ValueError, match="managed domain"):
        ranges.add(Span(1, 4))
    assert ranges.snapshot() == before
