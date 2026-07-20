# CDN byte-range cache

`CDNByteRangeCache` tracks residency for one immutable object byte domain.
Resident `CachedSegment` values are exact, non-overlapping ranges with stable
cache keys. Unallocated ranges represent cache misses rather than available
memory, which makes the shared allocator's complement directly useful for
request classification.

## Residency and coverage

`cache_segment(start, length, cache_key=...)` installs an exact segment. A key
replay with the same request is idempotent; a conflicting overlap or changed
request fails atomically. `evict(segment)` requires the exact live identity,
restores its bytes to missing geometry, and increments the eviction count.

`request_coverage(start, length)` clips both resident and missing ranges to the
requested byte interval. It reports normalized ranges, total covered bytes,
and `fully_resident`; requests must remain inside the object. This distinction
supports partial HTTP range responses without pretending that a segment-level
hit covers uncached gaps.

Snapshots expose all segments, object-wide missing ranges, resident bytes,
evictions, and free-range fragmentation. A checkpoint contains allocator
geometry, cache-key metadata, and counters. Restore validates a one-to-one
mapping to live allocator handles before changing state.

```python
from treemendous.applications.allocation.cdn_cache import CDNByteRangeCache

cache = CDNByteRangeCache(1_000_000, object_id="movie")
segment = cache.cache_segment(100_000, 200_000, cache_key="edge-a")
coverage = cache.request_coverage(250_000, 100_000)
print(coverage.resident_ranges, coverage.missing_ranges)
cache.evict(segment)
```

The executable example follows this lifecycle. The independent oracle classifies
each requested byte without production code. The smoke performs real residency,
coverage, and eviction operations.
