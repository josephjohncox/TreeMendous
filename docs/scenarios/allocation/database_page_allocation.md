# Database page allocation

`DatabasePageAllocator` models a named tablespace whose unit is a database
page. Leading metadata pages are permanently reserved. Each `PageExtent` is a
contiguous first-fit allocation owned by a table identity. First fit is
intentional: a recently freed low page range is reused before the high-water
free tail, matching the recycling behavior expected from a simple storage
engine free-space map.

## Reuse and observability

`allocate_pages(table_id, page_count)` returns an extent marked `reused=True`
when any of its pages intersect the normalized history of freed pages.
`free_pages` requires the exact live extent and matching table ID. Freed history
is retained independently of current free geometry, so diagnostics can report
`pages_ever_freed` and `reuse_allocations`. This is historical telemetry, not a
claim that those pages are still free.

Snapshots expose current table extents, free page spans, allocator
fragmentation, and reuse counters. Checkpoints preserve allocator state, extent
metadata, normalized freed history, and counters. Restore validates all of
these before replacing live state. Invalid page counts, ownership errors, and
capacity failures are non-mutating.

```python
from treemendous.applications.allocation.database_pages import DatabasePageAllocator

pages = DatabasePageAllocator(10000, tablespace="orders", metadata_pages=8)
staging = pages.allocate_pages("staging", 16)
pages.free_pages(staging, table_id="staging")
orders = pages.allocate_pages("orders", 8)
assert orders.reused
```

See `examples/applications/allocation/database_page_allocation.py`. The oracle
uses a standalone page bitmap. The smoke executes actual allocate/free cycles
and therefore exercises page-history normalization and reuse classification.
