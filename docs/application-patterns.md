# Application patterns

Tree-Mendous exposes reusable primitives and 50 concrete process-local engines.
A pattern shows how a primitive can fit inside a larger design; it is not a new
registered engine and does not add the surrounding service guarantees.

## Atomic geometry reconciliation

Use `ExactBatchRangeSet` when a controller has already decided which integer
spans to add and remove, and the geometry changes must become visible together.
The
[atomic port-pool reconciliation example](../examples/patterns/atomic_port_pool_reconciliation.py)
removes two reserved port bands in one ordered transaction and reports the new
free geometry.

This pattern provides:

- exact half-open integer geometry;
- ordered add/discard rows;
- whole-batch atomic publication or rollback;
- exact per-row changed spans;
- explicit resource limits.

It deliberately does **not** provide:

- lease or allocation identity;
- owner authentication or authorization;
- TTLs, clocks, renewal, expiry, or reclamation;
- fencing tokens or enforcement at a protected service;
- durable storage, replication, consensus, or cross-process coordination;
- `RangeSet` payload policies, allocation, or generic query methods.

Use the concrete TCP/UDP port lease engine when its lease lifecycle and local
fencing contract match the task. Use an external durable coordinator when
multiple processes must agree. Exact batch is not integrated into that engine
or any of the other 49 registered engines.

## Spatiotemporal overlap

Represent a rectangular spatial region active during a time window as a 3D
half-open box `(x, y, time)`. The
[spatiotemporal geofence example](../examples/patterns/spatiotemporal_geofences.py)
uses `BoxIndex3D` to retain record identity and query an observation volume.
Duplicate boxes remain separate records because removal and update use
owner-scoped handles.

This pattern is appropriate for bounded integer coordinate systems where an
axis-aligned box is a truthful approximation. It provides process-local insert,
update, exact handle removal, deterministic overlap order, and immutable
snapshots.

It deliberately does **not** provide:

- stable API status or a compatibility commitment;
- geographic coordinate reference systems, antimeridian handling, spherical
  distance, polygons, or uncertain coordinates;
- durable or distributed indexing;
- cross-process handles or authorization capabilities;
- an automatic choice between linear, projection, and sparse-grid strategies;
- registration of this `BoxIndex3D` pattern as an application engine.

The experimental index must be qualified against the target box distribution,
query selectivity, payload-copy cost, and process memory budget.

## Existing application-specific multidimensional semantics

Two registered engines have multidimensional domain meaning, but only one uses
the generic multidimensional index:

- **Radio spectrum timeslots** wrap an experimental `BoxIndex(2)` behind a
  scenario-specific scheduler. The engine adds channel bounds, guard bands,
  reservation identity, idempotency, cancellation, and application snapshots;
  callers do not manipulate its internal handles.
- **Morton-code geospatial ranges** do not use `BoxIndex`. They encode
  two-dimensional cells into one-dimensional Morton bands for candidate
  selection, then apply exact Cartesian filtering.

The new `BoxIndex3D` prototype is outside the registry and does not inherit the
radio scheduler's reservation semantics or the Morton catalog's filtering
contract.

## Combining primitives safely

If an application owns both a `RangeSet` and a `BoxIndex`, or an external
record table and an `ExactBatchRangeSet`, no cross-object transaction is
provided. Define publication order, rollback, recovery, and identity mapping in
the application. A snapshot from one object is not a snapshot of the combined
system.

Before adopting a pattern:

1. list the semantics intentionally kept outside the Tree-Mendous primitive;
2. assign one authority for identity and durable ownership;
3. define failure and replay behavior;
4. test half-open boundary cases and duplicate records;
5. benchmark the public application operation, not only geometry calls.

See [Choosing an interface](choosing-an-interface.md),
[Applications](applications.md), and [Performance](performance.md).
