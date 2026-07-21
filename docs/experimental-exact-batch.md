# Experimental exact whole-batch CPU mutations

`treemendous.experimental.exact_batch` is an opt-in experiment, not a stable
backend. It is absent from the package root, `RangeSetProtocol`, backend adapters,
capabilities, and the backend catalog. Compatibility and continued inclusion are
not guaranteed.

## API and denotation

```python
from array import array
from treemendous.experimental.exact_batch import ExactBatchRangeSet

ranges = ExactBatchRangeSet([(0, 100), (200, 300)], initially_available=False)
results = ranges.mutate_packed(array("q", [0, 10, 20, 1, 12, 14]))
```

`ExactBatchRangeSet` owns independent sorted-vector geometry. It requires a
non-empty managed domain, normalizes half-open domain spans by the canonical
`ManagedDomain` rules, and limits all endpoints and signed measures to signed
64-bit range. It has no payload policy, allocator, or generic query API.

`mutate_packed` accepts only a C-contiguous buffer of native-endian signed-int64
elements. Its shape must be flat `(3*N,)` or two-dimensional `(N, 3)`. Lists and
tuples are deliberately not hot-path inputs. Each row is `(opcode, start, end)`:

- `MutationOpcode.ADD == 0`
- `MutationOpcode.DISCARD == 1`
- `MutationOpcode.DISCARD_REQUIRE_COVERED == 2`

Rows execute sequentially in their supplied order. `fully_covered` describes the
pre-row available geometry. A strict discard changes nothing unless that value is
true. Duplicate and no-op rows remain in the result. Changed spans are the exact,
canonical, ascending geometry altered by that row; operations are never reordered.
Invalid or empty spans, unknown opcodes, spans outside or crossing one normalized
domain component, and signed measure/aggregate overflow identify the failing row.

The entire call is atomic. A nonblocking same-instance mutation guard is acquired
before invoking the buffer exporter. The input is copied while the GIL is held;
state is staged and rows execute natively with the GIL released. Packed storage and
its Python owner are prepared before an allocation-free `noexcept` state swap.
Every failure leaves both geometry and total unchanged. Snapshots synchronize with
the commit and therefore publish a complete pre- or post-batch state. Reentrant or
overlapping mutation of one instance raises `RuntimeError`; different instances
can execute concurrently.

`snapshot()` returns the package's exact canonical `RangeSnapshot`, including its
canonical `IntervalResult` tuple, exact total, and normalized domain.

## Packed results

`PackedMutationResults` owns immutable byte storage. Every property returns a
read-only native-endian buffer and remains valid after later manager mutations or
after the manager and result owner are destroyed:

| property | format | shape | meaning |
| --- | --- | --- | --- |
| `changed_offsets` | uint64 (`Q`) | `(N+1,)` | CSR row offsets |
| `changed_spans` | int64 (`q`) | `(M, 2)` | ascending `(start, end)` pairs |
| `changed_lengths` | int64 (`q`) | `(N,)` | exact changed measure per row |
| `fully_covered` | uint8 (`B`) | `(N,)` | zero/one pre-row coverage |

The CSR invariant is `offsets[0] == 0`, offsets are nondecreasing, and
`offsets[N] == M`. `len(results) == N`. `materialize()` constructs canonical
`Span` and `MutationResult` objects. It is intentionally outside the packed timed
path.

## Performance scope and promotion gates

`tests/performance/exact_batch_benchmark.py` uses deterministic per-size traces
starting from 64 disjoint available intervals. Every trace is validated outside
timing, row by row, against canonical public `cpp_boundary`, and every timed
repetition starts and ends in the exact same 64-interval state. The batch-2 trace
is a real mutation/restoration pair. Sizes 4 through 64 are composed of
restorative four-operation blocks with varied components; together their primary
workloads cover real restoration, duplicates, no-ops, strict rejections, partial
overlap, and changes spanning multiple live-state components. Batch 1 is a
separately labelled, state-preserving no-op call-overhead diagnostic.

The report identifies `cpp_boundary` explicitly as its baseline. If that compiled
backend is unavailable, the benchmark is declined; it never substitutes a slower
Python backend. Packed timings include buffer acquisition and copy, live-state
staging, ordered execution, packed allocation, atomic commit, and packed-result
destruction. They exclude setup, validation/invariant snapshots, and `materialize`;
materialization is reported separately. Ordinary runs are diagnostic:

```console
python tests/performance/exact_batch_benchmark.py --output report.json
```

Absolute gates are explicitly callable with `--enforce-hard-gates`. Promotion
requires all of the following without lowering thresholds:

- batch-16 throughput lower 95% bound at least 2 million logical operations/s;
- batch-16 speedup lower 95% bound at least 2x stable scalar;
- break-even lower 95% bound by batch size 4;
- no more than 3% stable scalar regression, established independently with the
  existing `mutation_attribution.py` baseline/candidate framework.

Sorted-vector staging is intentional, not a generic abstraction. Its cost scales
with live interval count on every batch and therefore has a vector-scaling cliff:
small batches or substantially larger fragmented states can erase the benefit.
Promotion requires evidence on the stated 64-interval workload and explicit
characterization of that cliff; this experiment is not a claim that vector state
is universally faster.
