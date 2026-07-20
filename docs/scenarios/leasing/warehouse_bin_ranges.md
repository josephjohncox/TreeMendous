# Warehouse bin ranges

`WarehouseBinPool` normalizes physical bin identifiers into numeric ranges while retaining operational compatibility metadata. Each named `BinZone` declares inclusive bin bounds, allowed size classes, and allowed hazard classes. A request must satisfy both label sets before capacity is inspected. This prevents a fallback allocator from silently placing oversized or hazardous inventory in an incompatible zone.

```python
zone = BinZone(
    500, 599,
    size_classes=frozenset({"large"}),
    hazards=frozenset({"flammable", "general"}),
)
lease = bins.acquire(
    "chemical", "putaway-17", ttl=20, count=4,
    size_class="large", hazard="flammable", request_id="task-17",
)
```

The returned `BinLease` retains the selected size and hazard labels alongside its numeric lease. Named zones use independent `LeasePool` instances; overlapping normalized IDs in different zones are therefore valid. Within one zone, allocation is earliest-first and contiguous. An exact `start_bin` is available when a warehouse control system has already selected a physical run.

Request idempotency includes zone, owner, TTL, count, exact span, size class, and hazard. Reusing an ID with changed compatibility metadata or zone produces `BinRequestConflictError`; it cannot accidentally return a semantically different bin lease. Renewal preserves compatibility labels. Release and expiry restore the range. Domain snapshots join every lease with its metadata, while diagnostics remain grouped by zone. Checkpoints preserve zones, token-scoped metadata, request evidence, and pool histories.

`validate_fence(lease, bin_id)` uses the stable key `("warehouse-bin-ranges", zone, bin_id)`. It validates membership before updating the high-water mark. Per-bin keys are important: a later job may request a differently sized range that overlaps a previous job, yet the protected physical bin still has one fencing history.

The engine does not model aisle travel, weight aggregation, physical locks, or a distributed warehouse database. Compatibility policy is fixed when the engine is built. All state transitions and the included validator are local to one process. A stale robot or worker is not stopped by TTL expiry. The warehouse execution system must persist the highest token and reject stale work at the bin operation. Checkpoint restore starts fresh pool lineages only after an external single-writer handoff, and fence high-water state must be restored by the downstream system separately.

Run the executable example with:

```console
uv run python examples/applications/leasing/warehouse_bin_ranges.py
```
