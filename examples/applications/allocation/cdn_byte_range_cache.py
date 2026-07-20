"""Query CDN request coverage and evict a byte segment."""

from treemendous.applications.allocation.cdn_cache import CDNByteRangeCache


def main() -> None:
    cache = CDNByteRangeCache(1_000_000, object_id="movie")
    segment = cache.cache_segment(100_000, 200_000, cache_key="edge-1")
    coverage = cache.request_coverage(250_000, 100_000)
    print("resident", coverage.resident_ranges, "missing", coverage.missing_ranges)
    cache.evict(segment)
    print("evictions", cache.snapshot().diagnostics.evictions)


if __name__ == "__main__":
    main()
