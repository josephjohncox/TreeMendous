"""CDN byte-range cache application contracts."""

from tests.oracles.applications.allocation.cdn_byte_range_cache import coverage
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
