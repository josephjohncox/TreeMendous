# Examples

## Concrete application engines

The repository has one executable example for each of the 50 registered
process-local engines. Browse them by family:

- [Partitioning and work claiming](applications/partitioning/), with 12 search,
  scan, sharding, and finite-work engines
- [Scheduling and reservation](applications/scheduling/), with 12
  cumulative-capacity and domain reservation engines
- [Identity-preserving overlap catalogs](applications/catalogs/), with 10
  record, coverage, lock, and reassembly engines
- [Allocation and capacity tracking](applications/allocation/), with 8
  allocator, cache, upload, and sequence engines
- [Numeric resource leasing](applications/leasing/), with 8 lifecycle and
  fencing-aware lease engines

The [application documentation index](../docs/applications.md) links each
example to its scenario contract. Run every application example plus the stable
and experimental examples below from an unrelated working directory with:

```bash
just run-examples
```

Run one example directly with the installed project environment:

```bash
uv run python examples/applications/partitioning/document_search.py
```

## Stable one-dimensional API

[`basic_rangeset.py`](basic_rangeset.py) uses only the stable public `RangeSet`
API:

```bash
uv run python examples/basic_rangeset.py
```

## Stable exact-batch API

[`exact_batch.py`](exact_batch.py) demonstrates ergonomic ordered mutations,
exact per-row results, explicit resource limits, and restoration of the initial
snapshot:

```bash
uv run python examples/exact_batch.py
```

## Reusable patterns outside the engine registry

These examples are independently executable patterns, not additions to the 50
registered engines:

- [`patterns/atomic_port_pool_reconciliation.py`](patterns/atomic_port_pool_reconciliation.py)
  uses `ExactBatchRangeSet` for one geometry-only transaction. It has no lease
  identity, TTL, fencing, persistence, or cross-process coordination.
- [`patterns/spatiotemporal_geofences.py`](patterns/spatiotemporal_geofences.py)
  uses experimental, process-local `BoxIndex3D`. It supplies no geographic
  coordinate system, durability, distribution, or authorization.

```bash
uv run python examples/patterns/atomic_port_pool_reconciliation.py
uv run python examples/patterns/spatiotemporal_geofences.py
```

See [Application patterns](../docs/application-patterns.md) for the surrounding
system responsibilities and exclusions.

## Experimental multidimensional API

[`multidimensional/core/linear_box_index.py`](multidimensional/core/linear_box_index.py)
demonstrates duplicate record identity, deterministic overlap order, update,
and exact handle removal:

```bash
uv run python examples/multidimensional/core/linear_box_index.py
```

The registered radio-spectrum engine wraps experimental `BoxIndex(2)` behind
its application-specific channel/time reservation contract. The Morton
geospatial example does not use `BoxIndex`; it uses one-dimensional Morton
bands plus exact Cartesian filtering. The new `BoxIndex3D` pattern remains
outside the 50-engine registry.

Backend implementation modules remain internal. Applications should construct
one-dimensional range sets through `treemendous.create_range_set`, import the
application registry through `treemendous.applications`, and import
experimental multidimensional values explicitly from
`treemendous.multidimensional`.
