# Benchmarking

Benchmarks are directional engineering measurements, not pull-request gates or
published speedup guarantees. Run the common correctness-checked harness with:

```bash
just benchmark
```

The harness generates one immutable operation trace and replays the identical
order and semantics for each backend. `execution` measures only that declared
public-operation replay. A separate equivalent replay performs per-operation
instrumentation, accounting, snapshots, normalization, JSON serialization,
checksums, and divergence rejection; its overhead is reported separately and is
never included in `execution`. Every query and final checksum is checked against
a benchmark-local ordered-list oracle with independent value types and
validators before timing is accepted. Reports distinguish requested operations,
successful mutations, no-ops, errors, coordinate extent, actual interval count,
touched intervals, and checksums.

For meaningful results, use multiple independent processes and samples, randomize
backend order, retain environment/commit/compiler/device metadata, and report
medians with intervals rather than a single run. Occupancy, fragmentation,
update/query ratio, fit position, impossible fits, and device upload/query ratio
must be explicit. Memory figures require measured RSS/allocation data; fabricated
per-node constants are not valid.

Accelerator entry points remain experimental and report unavailability instead
of silently omitting a backend. Scheduled/manual CI may archive machine-readable
samples. PR CI has no wall-clock benchmark gate.
