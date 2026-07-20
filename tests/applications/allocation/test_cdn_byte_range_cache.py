"""CDN byte-range cache application contracts."""

from dataclasses import replace
from typing import cast

import pytest

from tests.oracles.applications.allocation.cdn_byte_range_cache import coverage
from treemendous.applications._shared.allocation import AllocationConflictError
from treemendous.applications.allocation.cdn_cache import CDNByteRangeCache
from treemendous.domain import Span


def test_request_coverage_matches_independent_byte_oracle() -> None:
    cache = CDNByteRangeCache(100, object_id="video")
    first = cache.cache_segment(10, 20, cache_key="a")
    cache.cache_segment(40, 10, cache_key="b")
    result = cache.request_coverage(20, 30)
    expected_bytes, expected_missing = coverage(
        Span(20, 50), (Span(10, 30), Span(40, 50))
    )
    assert result.covered_bytes == expected_bytes
    assert result.missing_ranges == expected_missing
    assert not result.fully_resident
    assert first.byte_range == Span(10, 30)


def test_eviction_changes_residency_and_checkpoint_restores() -> None:
    cache = CDNByteRangeCache(40)
    segment = cache.cache_segment(0, 20, cache_key="prefix")
    checkpoint = cache.checkpoint()
    cache.evict(segment)
    missing = cache.request_coverage(0, 20).missing_ranges
    assert len(missing) == 1
    assert missing[0] == Span(0, 20)
    cache.restore(checkpoint)
    snapshot = cache.snapshot()
    assert snapshot.diagnostics.resident_bytes == 20
    assert snapshot.diagnostics.evictions == 0


def test_adjacent_segments_normalize_whole_object_coverage() -> None:
    cache = CDNByteRangeCache(20)
    cache.cache_segment(0, 10, cache_key="first")
    cache.cache_segment(10, 10, cache_key="second")

    coverage = cache.request_coverage(0, 20)

    assert coverage.resident_ranges == (Span(0, 20),)
    assert coverage.missing_ranges == ()
    assert coverage.covered_bytes == 20
    assert coverage.fully_resident


def test_cache_key_replay_and_conflict_are_typed_and_atomic() -> None:
    cache = CDNByteRangeCache(20)
    segment = cache.cache_segment(0, 10, cache_key="key")
    assert cache.cache_segment(0, 10, cache_key="key") is segment
    before = cache.snapshot()

    with pytest.raises(AllocationConflictError):
        cache.cache_segment(10, 10, cache_key="key")
    assert cache.snapshot() == before

    with pytest.raises(ValueError, match="nonempty string"):
        cache.cache_segment(10, 10, cache_key=1)  # type: ignore[arg-type]
    assert cache.snapshot() == before


def test_restore_rejects_forged_reserved_geometry_atomically() -> None:
    cache = CDNByteRangeCache(20)
    checkpoint = cache.checkpoint()
    forged_allocator = replace(
        checkpoint.allocator,
        reserved_ranges=(Span(0, 5),),
        free_ranges=(Span(5, 20),),
    )
    before = cache.snapshot()

    with pytest.raises(ValueError, match="configured allocator geometry"):
        cache.restore(replace(checkpoint, allocator=forged_allocator))

    assert cache.snapshot() == before


def test_restore_rejects_non_string_cache_key_atomically() -> None:
    cache = CDNByteRangeCache(20, object_id="object")
    cache.cache_segment(0, 10, cache_key="key")
    checkpoint = cache.checkpoint()
    bad_key = cast(str, 7)
    record = replace(checkpoint.allocator.records[0], idempotency_key=bad_key)
    forged_allocator = replace(
        checkpoint.allocator,
        records=(record,),
        idempotency=(("object", bad_key, record.handle.allocation_id),),
    )
    segment = replace(checkpoint.segments[0], cache_key=bad_key)
    before = cache.snapshot()

    with pytest.raises(ValueError, match="cached-segment metadata"):
        cache.restore(
            replace(checkpoint, allocator=forged_allocator, segments=(segment,))
        )

    assert cache.snapshot() == before
