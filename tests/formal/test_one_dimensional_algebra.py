"""Exhaustive finite-domain checks for the one-dimensional formal model."""

from __future__ import annotations

from itertools import product

from tests.formal.one_dimensional_model import (
    all_point_sets,
    denotation,
    measure,
    normalize,
    points,
    release,
    release_absorbed_count,
    reserve,
    reserve_structure,
    runs,
    valid_spans,
)
from treemendous.domain import AvailabilityStats

EXTENT = 6


def test_normal_form_exists_is_unique_and_is_idempotent() -> None:
    universe = all_point_sets(EXTENT)
    for free in universe:
        canonical = runs(free)
        assert denotation(canonical) == free
        assert normalize(canonical) == canonical
        assert all(start < end for start, end in canonical)
        assert all(
            left_end < right_start
            for (_, left_end), (right_start, _) in zip(canonical, canonical[1:])
        )

    # Every two-span presentation refines to the same tuple as its point set.
    presentations = ((), *((span,) for span in valid_spans(EXTENT)))
    for left, right in product(presentations, repeat=2):
        presentation = (*left, *right)
        canonical = normalize(presentation)
        assert canonical == runs(denotation(presentation))


def test_release_and_reserve_algebra_measure_and_locality() -> None:
    for free in all_point_sets(EXTENT):
        canonical_count = len(runs(free))
        for target in valid_spans(EXTENT):
            target_points = points(target)

            released = release(free, target)
            release_changed = target_points - free
            assert released.after == free | target_points
            assert denotation(released.changed) == release_changed
            assert released.changed_length == measure(release_changed)
            assert released.fully_covered is (target_points <= free)
            assert measure(released.after) == measure(free) + measure(release_changed)
            assert (released.after ^ free) <= target_points
            assert measure(target_points - free) + measure(
                target_points & free
            ) == measure(target_points)

            if release_changed:
                absorbed = release_absorbed_count(free, target)
                after_count = len(runs(released.after))
                assert after_count == canonical_count - absorbed + 1
                concrete_splice_work = absorbed + 1
                potential_change = after_count - canonical_count
                assert concrete_splice_work + potential_change == 2
                assert after_count <= canonical_count + 1

            reserved = reserve(free, target)
            reserve_changed = target_points & free
            assert reserved.after == free - target_points
            assert denotation(reserved.changed) == reserve_changed
            assert reserved.changed_length == measure(reserve_changed)
            assert reserved.fully_covered is (target_points <= free)
            assert measure(reserved.after) == measure(free) - measure(reserve_changed)
            assert (reserved.after ^ free) <= target_points

            if reserve_changed:
                affected, remainders = reserve_structure(free, target)
                after_count = len(runs(reserved.after))
                assert after_count == canonical_count - affected + remainders
                concrete_splice_work = affected + remainders
                potential_change = after_count - canonical_count
                assert concrete_splice_work + potential_change == 2 * remainders
                assert concrete_splice_work + potential_change <= 4
                assert after_count <= canonical_count + 1


def test_require_covered_rejection_is_an_atomic_observable_noop() -> None:
    for free in all_point_sets(EXTENT):
        for target in valid_spans(EXTENT):
            target_points = points(target)
            result = reserve(free, target, require_covered=True)
            if target_points <= free:
                assert result == reserve(free, target)
            else:
                assert result.after == free
                assert not result.changed
                assert result.changed_length == 0
                assert not result.fully_covered


def test_fragmentation_avoids_cancellation_near_one() -> None:
    largest = 1 << 54
    stats = AvailabilityStats(
        total_free=largest + 1,
        free_chunks=2,
        largest_chunk=largest,
    )
    assert stats.fragmentation > 0.0
    assert stats.fragmentation == 1 / (largest + 1)


def test_fragmentation_measure_bounds_hold_for_every_nonempty_state() -> None:
    for free in all_point_sets(EXTENT):
        canonical = runs(free)
        if not canonical:
            continue
        count = len(canonical)
        total = measure(free)
        largest = max(end - start for start, end in canonical)
        assert total / count <= largest <= total
        fragmentation = 1.0 - largest / total
        assert 0.0 <= fragmentation <= 1.0 - 1.0 / count
        assert (fragmentation == 0.0) is (count == 1)
