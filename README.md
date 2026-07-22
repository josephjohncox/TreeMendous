# Tree-Mendous

[![PyPI](https://img.shields.io/pypi/v/treemendous.svg)](https://pypi.org/project/treemendous/)
[![Python](https://img.shields.io/pypi/pyversions/treemendous.svg)](https://pypi.org/project/treemendous/)
[![CI](https://github.com/josephjohncox/TreeMendous/actions/workflows/ci-cd.yaml/badge.svg)](https://github.com/josephjohncox/TreeMendous/actions/workflows/ci-cd.yaml)
[![License](https://img.shields.io/github/license/josephjohncox/TreeMendous.svg)](https://github.com/josephjohncox/TreeMendous/blob/main/LICENSE)

Tree-Mendous provides exact half-open integer range sets: `[start, end)` includes
`start` and excludes `end`. Choose the stable `RangeSet` interface for payloads,
queries, and allocation; a specialized native API for atomic geometry batches;
50 process-local application engines; or explicitly experimental
multidimensional indexes.

## Choose an API

| Need | Interface | Status | Important boundary |
| --- | --- | --- | --- |
| Add, discard, query, allocate, snapshot, or attach payloads | `RangeSet` via `treemendous.create_range_set` | Stable | The canonical general-purpose API |
| Apply ordered geometry mutations as one native transaction | `treemendous.exact_batch` | Stable, specialized | No payloads, allocation, or generic query API |
| Use a concrete scheduler, allocator, catalog, partitioner, or lease pool | `treemendous.applications` | Stable application namespace | 50 process-local engines with scenario-specific contracts |
| Index identity-preserving 2D–4D boxes | `treemendous.multidimensional` | Experimental | Process-local; not exported from the package root |

The [task-oriented interface guide](https://github.com/josephjohncox/TreeMendous/blob/main/docs/choosing-an-interface.md)
explains these boundaries in detail.

## Install

Tree-Mendous supports CPython 3.11–3.13.

```bash
python -m pip install treemendous
```

## `RangeSet` quickstart

This complete example reserves unavailable edges, atomically allocates the
first two-unit slot, and checks the remaining capacity:

```python
from treemendous import Span, create_range_set

ranges = create_range_set((0, 24), backend="py_boundary")
ranges.discard(Span(0, 9))
ranges.discard(Span(17, 24))

booking = ranges.allocate(2, not_before=9, not_after=17)
assert booking is not None
assert booking.span == Span(9, 11)
assert ranges.first_fit(2, not_before=11, not_after=17).span == Span(11, 13)
assert ranges.snapshot().total_free == 6
```

`allocate` returns `None` if no fit exists. Coordinates must be integers,
lengths must be positive, and mutations must stay inside the managed domain.
Payload behavior is selected with an explicit `UniformPayloadPolicy`,
`JoinPayloadPolicy`, or `OrderedPayloadPolicy`; see the
[API guide](https://github.com/josephjohncox/TreeMendous/blob/main/docs/api.md).

## Exact-batch quickstart

Use `ExactBatchRangeSet` when one ordered, geometry-only batch must either
publish completely or leave the prior snapshot visible:

```python
from treemendous import Span
from treemendous.exact_batch import (
    BatchMutation,
    ExactBatchRangeSet,
    MutationOpcode,
)

ranges = ExactBatchRangeSet((0, 64), initially_available=False)
results = ranges.mutate(
    [
        BatchMutation(MutationOpcode.ADD, 8, 20),
        BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 10, 14),
    ]
)

assert [result.changed for result in results] == [
    (Span(8, 20),),
    (Span(10, 14),),
]
assert ranges.snapshot().intervals[0].span == Span(8, 10)
```

Rows execute in input order. Per-instance limits bound operations, live
intervals, changed spans, packed-result bytes, and staging work. Exact batch is
not a `RangeSet` backend and is not integrated into the 50 application engines.
Read the [exact-batch contract](https://github.com/josephjohncox/TreeMendous/blob/main/docs/exact-batch.md).

## Performance

Performance depends on workload shape, state size, backend availability,
platform, and which wrapper or materialization layers are timed. Current local
Apple M5 Max/macOS 26.5.1/CPython 3.12.7 standard measurements put
`cpp_boundary` at about 0.30M–0.91M timed public operations per second across
six traces with 64–128 initial intervals. This is evidence for those traces,
not a universal throughput claim. Large snapshots and sorted-vector exact
batches can become
copy-bound.

The [performance guide](https://github.com/josephjohncox/TreeMendous/blob/main/docs/performance.md)
reports operation-level measurements, hosted 1.1.0 exact-batch evidence, timing
boundaries, and the optimization roadmap. The
[benchmark methodology](https://github.com/josephjohncox/TreeMendous/blob/main/docs/benchmarking.md)
explains correctness checks and durable artifacts.

## Applications and reusable patterns

The 50 concrete engines cover partitioning, scheduling, overlap catalogs,
allocation, and numeric leasing. Each has its own factory, state model, and
exclusions; names describing distributed work do not imply transport,
consensus, or durable storage. Start with the
[application index](https://github.com/josephjohncox/TreeMendous/blob/main/docs/applications.md)
or the [application-pattern guide](https://github.com/josephjohncox/TreeMendous/blob/main/docs/application-patterns.md).

Two additional executable patterns demonstrate APIs outside the 50-engine
registry:

- [atomic port-pool reconciliation](https://github.com/josephjohncox/TreeMendous/blob/main/examples/patterns/atomic_port_pool_reconciliation.py)
  uses exact batch for geometry only; it has no lease identity, TTL, or fencing;
- [spatiotemporal geofences](https://github.com/josephjohncox/TreeMendous/blob/main/examples/patterns/spatiotemporal_geofences.py)
  uses experimental, process-local `BoxIndex3D`.

The registered radio-spectrum engine is the existing generic `BoxIndex`
integration: it wraps an experimental `BoxIndex(2)` inside its stable,
application-specific reservation contract. The Morton catalog instead uses
one-dimensional Morton candidate bands plus exact Cartesian filtering. The new
`BoxIndex3D` pattern is outside the registry, and exact batch is not integrated
into any of the 50 engines.

## Backend maturity

Automatic selection considers only stable backends that are available, pass
semantic probes, and satisfy requested capabilities. Selecting an unavailable
or invalid backend raises a reasoned error.

| Backend ID | Runtime | Width | Maturity | Notes |
| --- | ---: | ---: | --- | --- |
| `py_boundary` | Python/CPU | 64-bit | Stable | Core geometry |
| `py_avl_earliest` | Python/CPU | 64-bit | Stable | Core geometry |
| `py_summary` | Python/CPU | 64-bit | Stable | Best-fit + analytics |
| `py_treap` | Python/CPU | 64-bit | Stable | Random interval sampling |
| `py_boundary_summary` | Python/CPU | 64-bit | Stable | Best-fit + analytics |
| `cpp_boundary` | C++/CPU | 64-bit | Stable when built | Core geometry |
| `cpp_treap` | C++/CPU | 32-bit | Experimental | Not selectable |
| `cpp_boundary_summary` | C++/CPU | 32-bit | Experimental | Not selectable |
| `cpp_boundary_summary_optimized` | C++/CPU | 32-bit | Experimental | Not selectable |
| `gpu_boundary_summary` | CUDA/GPU | 32-bit | Experimental | Not selectable |
| `metal_boundary_summary` | Metal/GPU | 32-bit | Experimental | Not selectable |

See the [backend catalog](https://github.com/josephjohncox/TreeMendous/blob/main/docs/backends.md)
for discovery and qualification details.

## Documentation

- [Documentation index](https://github.com/josephjohncox/TreeMendous/blob/main/docs/README.md)
- [Getting started](https://github.com/josephjohncox/TreeMendous/blob/main/docs/getting-started.md)
- [Choosing an interface](https://github.com/josephjohncox/TreeMendous/blob/main/docs/choosing-an-interface.md)
- [Canonical API](https://github.com/josephjohncox/TreeMendous/blob/main/docs/api.md)
- [Performance](https://github.com/josephjohncox/TreeMendous/blob/main/docs/performance.md)
- [Examples](https://github.com/josephjohncox/TreeMendous/blob/main/examples/README.md)
- [Release notes](https://github.com/josephjohncox/TreeMendous/blob/main/docs/releases/1.1.1.md)

## Development

```bash
git clone https://github.com/josephjohncox/TreeMendous.git
cd TreeMendous
uv sync --all-extras
just check
just run-examples
```

Use `just build` for locally verified wheel and source artifacts. See
[Contributing](https://github.com/josephjohncox/TreeMendous/blob/main/docs/contributing.md)
and [Releasing](https://github.com/josephjohncox/TreeMendous/blob/main/docs/releasing.md)
for the complete quality and publication contracts.

BSD-3-Clause license.
