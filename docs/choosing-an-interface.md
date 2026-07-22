# Choosing an interface

Start with the behavior the caller must preserve. The interfaces overlap in
geometry vocabulary, but they are not interchangeable backends for one object.

## Decision map

- **Maintain, query, or allocate available integer spans:** choose `RangeSet`
  through `treemendous.create_range_set`. It is the stable canonical API for
  managed domains, allocation, snapshots, statistics, and payload policies. It
  does not provide one transaction spanning multiple public calls.
- **Apply ordered add/discard rows atomically:** choose
  `treemendous.exact_batch.ExactBatchRangeSet`. It returns exact per-row results
  and enforces explicit limits, but it has no payload, allocation, generic
  query, or `RangeSetProtocol` support.
- **Use a registered scheduler, partitioner, catalog, allocator, or lease
  pool:** choose `treemendous.applications`. Each engine owns scenario-specific
  identity, lifecycle, validation, and results. None supplies durable or
  distributed infrastructure.
- **Store identity-preserving 2D–4D boxes and query overlap:** choose
  `treemendous.multidimensional` only as an explicit experiment. It provides
  box values, handles, overlap indexes, and snapshots, but no stable API,
  persistence, or cross-process coordination.

## Use `RangeSet` by default

Choose `RangeSet` for general half-open integer range work:

- `add`, `discard`, `intervals`, `overlaps`, `first_fit`, and `allocate`;
- managed domains with disconnected components;
- snapshots and availability statistics;
- uniform, join, or ordered payload policies;
- stable backend discovery and qualification.

Construct it with `create_range_set`. Omit `backend` for stable automatic
selection, or request a stable backend explicitly for reproducibility. Raw
geometry implementation modules are not public application interfaces.

`RangeSet` mutations are individually atomic. There is no stable multi-call
transaction or mutate-many API in 1.1.1.

## Use exact batch only for geometry transactions

Choose `ExactBatchRangeSet` when all of these statements are true:

1. every row is an add, discard, or strict discard of integer geometry;
2. row order matters;
3. the whole batch must publish or roll back;
4. explicit staging and result limits are acceptable;
5. no payload, allocation, or generic query is required.

It owns independent sorted-vector state. It is not exported from the package
root, does not implement `RangeSetProtocol`, and cannot be selected by the
backend registry. It is also not an acceleration switch inside the 50
application engines. Keep lease identity, TTL, fencing, ownership, and durable
coordination in an appropriate application layer.

See [Exact batch](exact-batch.md) and the
[atomic port-pool pattern](../examples/patterns/atomic_port_pool_reconciliation.py).

## Use a concrete application engine for its exact scenario

The application namespace contains 50 process-local engines. Prefer one when
its documented contract already matches the task: for example, a fenced port
lease pool, a cumulative-capacity scheduler, or an identity-preserving interval
catalog. Import the registry from `treemendous.applications`, or import a
concrete class from its scenario module.

Read the scenario page before choosing. A descriptive name does not supply
network transport, consensus, durable persistence, external authorization, or
cross-process safety. Exact batch is separate and is not integrated into these
engines. The radio-spectrum scheduler internally uses experimental
`BoxIndex(2)`, but callers should depend on its application contract rather than
treating it as a generic index façade.

See the [application index](applications.md) and
[application patterns](application-patterns.md).

## Use multidimensional indexes as explicit experiments

Choose `BoxIndex2D`, `BoxIndex3D`, `BoxIndex4D`, or `BoundedBoxIndex` only when
axis-aligned half-open boxes and identity-preserving overlap queries match the
model. These indexes are experimental, process-local, and imported explicitly
from `treemendous.multidimensional`. They do not provide spatial reference
systems, spherical geometry, persistence, authorization, or distributed query
execution.

The radio-spectrum application engine wraps generic `BoxIndex(2)` with its own
channel/time, guard-band, identity, and idempotency rules. The Morton geospatial
catalog does not use `BoxIndex`; it uses one-dimensional Morton candidate bands
plus exact Cartesian filtering.

See the [box-index semantics](theory/box_index_denotation.md) and
[spatiotemporal pattern](../examples/patterns/spatiotemporal_geofences.py).

## Validate the decision

Before deployment, write down:

- coordinate units, bounds, and half-open conventions;
- payload, record identity, ownership, lease, and fencing requirements;
- transaction and rollback boundaries;
- persistence and process boundaries;
- expected interval or box count and operation mix;
- the exact public layer to benchmark.

Then test the chosen interface's semantic contract and benchmark the concrete
state shapes. The [performance guide](performance.md) explains why headline
throughput cannot substitute for that qualification.
