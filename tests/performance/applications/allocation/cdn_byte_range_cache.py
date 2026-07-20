"""Actual CDN residency/coverage/eviction smoke benchmark."""

from time import perf_counter

from treemendous.applications.allocation.cdn_cache import CDNByteRangeCache


def run_smoke(operations: int = 1000) -> dict[str, int | float]:
    if operations <= 0:
        raise ValueError("operations must be positive")
    cache = CDNByteRangeCache(4096)
    started = perf_counter()
    covered = 0
    for index in range(operations):
        segment = cache.cache_segment(0, 64, cache_key=str(index))
        covered += cache.request_coverage(0, 128).covered_bytes
        cache.evict(segment)
    return {
        "operations": operations,
        "covered_bytes": covered,
        "seconds": perf_counter() - started,
    }


if __name__ == "__main__":
    print(run_smoke())
