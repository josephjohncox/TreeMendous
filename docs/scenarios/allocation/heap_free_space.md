# Heap free-space allocation

`HeapAllocator` turns `ContiguousAllocator` into a guarded heap layout engine.
A request has four distinct regions: a header, a left redzone, the aligned
payload, and a right redzone. Only the payload address is alignment-sensitive;
alignment padding remains unavailable as part of the raw block only when it is
needed before the header. Alignment must be a positive power of two.

## Operations and invariants

`allocate(size, owner=..., alignment=..., policy=...)` supports deterministic
first, best, and worst fit. Policies rank complete free chunks, then choose the
lowest valid aligned address. `free(block, owner=...)` requires both the opaque
allocator handle and its owner, rejecting stale or cross-owner releases.
Headers and redzones are never exposed as payload. The snapshot reports raw
free ranges, payload layouts, and allocator fragmentation; allocated bytes in
diagnostics include overhead by design.

The engine validates a complete candidate before reserving it, so an alignment
or capacity failure is non-mutating. `checkpoint()` captures allocator geometry
and heap layouts. `restore()` validates every payload against its raw handle
before committing, and allocator handle counters are never reused after a
restore.

```python
from treemendous.applications.allocation.heap import HeapAllocator

heap = HeapAllocator(4096, header_size=8, redzone_size=16)
block = heap.allocate(100, owner="request", alignment=64, policy="best")
assert block.payload.start % 64 == 0
heap.free(block, owner="request")
```

Run `examples/applications/allocation/heap_free_space.py` for an executable
walkthrough. The smoke benchmark performs real guarded allocation/free churn;
the independent test oracle searches candidate byte addresses without using
the production allocator.
