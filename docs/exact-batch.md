# Stable exact whole-batch CPU mutations

`treemendous.exact_batch` is the stable specialized geometry-batch API. It is not
exported from the package root and is not a `RangeSetProtocol` backend. `RangeSet`
remains the sole stable root one-dimensional API.

## Stable contract

```python
from treemendous.exact_batch import (
    BatchMutation,
    ExactBatchRangeSet,
    MutationOpcode,
)

ranges = ExactBatchRangeSet([(0, 100), (200, 300)], initially_available=False)
results = ranges.mutate([
    BatchMutation(MutationOpcode.ADD, 10, 20),
    BatchMutation(MutationOpcode.DISCARD, 12, 14),
])
```

The public names are `ExactBatchRangeSet`, `MutationOpcode`, immutable
`BatchMutation`, immutable `BatchLimits`, `PackedMutationResults`, and
`BatchLimitError`. An exact-batch range set owns independent geometry. It has no
payloads, payload policy, allocation operation, or generic range-query API.
Its only operations are `mutate`, `mutate_packed`, `snapshot`, and `domain`.

A managed domain is non-empty and normalized by the canonical `ManagedDomain`
rules. All endpoints and aggregate measures must fit signed 64-bit integers.
Each mutation span is half-open and must lie inside one normalized domain
component. Rows execute in order and are never reordered:

- `MutationOpcode.ADD == 0`
- `MutationOpcode.DISCARD == 1`
- `MutationOpcode.DISCARD_REQUIRE_COVERED == 2`

`fully_covered` describes the geometry before its row. A strict discard is an
exact no-op unless that value is true. Duplicate and no-op rows remain in the
result. Changed spans are exact, canonical, ascending geometry altered by that
row. Invalid enum values, booleans used as integers, invalid or empty spans,
unknown packed opcodes, component-crossing spans, and overflow are rejected.

`mutate(iterable[BatchMutation])` is the ergonomic builder and returns a tuple of
canonical `MutationResult` values. It validates the immutable rows, encodes them
to immutable bytes, and uses the same native path as `mutate_packed`.

`mutate_packed` accepts **exact `bytes` only**. Neither bytes subclasses nor other
buffer exporters are accepted. The byte length must be a multiple of 24. Each row
is three native-endian signed 64-bit integers `(opcode, start, end)`. This ABI is
intentionally native-endian; callers persisting or transporting traces must encode
for the consuming machine. Immutable input prevents a writable-buffer race.

`snapshot()` returns the package's canonical `RangeSnapshot`, with exact
`IntervalResult` values, total, and normalized domain. Inherited live instances
are unsupported after a multithreaded process calls `fork`; construct a new
instance in the child.

## Limits

Every instance has checked `BatchLimits`. Values must be positive integers, must
not be booleans, and must fit Python's signed size range. Conservative defaults
are:

| limit | default |
| --- | ---: |
| `max_operations` | 1,000,000 |
| `max_live_intervals` | 100,000 |
| `max_changed_spans` | 2,000,000 |
| `max_result_bytes` | 256 MiB |
| `max_work_units` | 100,000,000 |

Work units are one row dispatch plus the number of live intervals presented to
that row. Result bytes are accounted exactly as `(N+1)*8 + M*16 + N*8 + N`, where
`N` is the row count and `M` is the cumulative changed-span count. A limit failure
raises `BatchLimitError`, a `ValueError` subclass. Input operation count is checked
before native operation-copy allocation. All other limits are checked while
staging and every failure leaves the published geometry and total unchanged.

## Packed results

`PackedMutationResults` owns immutable byte storage. Its properties are read-only
native-endian memoryviews and remain valid after later mutations or destruction of
the originating range set:

| property | format | shape | meaning |
| --- | --- | --- | --- |
| `changed_offsets` | uint64 (`Q`) | `(N+1,)` | CSR row offsets |
| `changed_spans` | int64 (`q`) | `(M, 2)` | ascending `(start, end)` pairs |
| `changed_lengths` | int64 (`q`) | `(N,)` | exact changed measure per row |
| `fully_covered` | uint8 (`B`) | `(N,)` | zero/one pre-row coverage |

Offsets begin at zero, are nondecreasing, and end at `M`. `len(results) == N`.
`materialize()` constructs canonical `Span` and `MutationResult` objects.

## Implementation and concurrency

The current C++20 implementation uses independent sorted-vector geometry. This is
an implementation choice, not a promise that vector state is universally faster.
A nonblocking per-instance guard rejects overlapping mutation with `RuntimeError`.
The implementation copies live state and total under the state mutex, releases the
mutex for scratch execution and complete result preparation, and reacquires it
only for an allocation-free commit. A snapshot can therefore proceed during a
long staged mutation and observes either the complete old state or complete new
state, never an intermediate state. Different instances can execute concurrently.
Python signals are checked periodically and interruption rolls back the batch.

Published wheels support CPython 3.11 through 3.13. Source distributions require a
C++20 compiler and pybind11. The extension opts into pybind11 3's per-interpreter
GIL support. This is not a free-threaded CPython support guarantee; no claim is
made for builds without the GIL.

## Benchmark scope

`tests/performance/exact_batch_benchmark.py` compares deterministic restorative
traces against the stable `cpp_boundary` backend and checks every row outside the
timed region. Its existing artifact schema remains unchanged for compatibility in
this phase. Ordinary diagnostic runs use:

```console
python -m tests.performance.exact_batch_benchmark \
  --output build/benchmarks/exact-batch.json
```

The original benchmark deliberately characterizes a 64-live-interval workload
and retains its fixed batch-4 break-even, batch-16 2x speedup, and batch-16
2,000,000-logical-operations/second gates.

`tests.performance.exact_batch_scaling` separately qualifies the production
sorted-vector envelope at 64, 1,000, 10,000, and 100,000 live intervals with a
batch size of 16. Each case starts with N disjoint managed-domain components
available. Restorative remove/re-add operations, strict rejections, and no-ops
near the end of the vector make every timed call begin and end with exactly N
intervals while exercising whole-state staging. The fixed production gate is
the upper bound of the bootstrap 95% confidence interval for median batch
latency at 100,000 intervals: no more than 10 ms. A clean candidate checkout
can generate and verify the canonical triplet with:

```console
python -m tests.performance.exact_batch_scaling --samples 20 \
  --output build/benchmarks/exact-batch-scaling.json --enforce-gate
python -m scripts.verify_exact_batch_scaling \
  build/benchmarks/exact-batch-scaling.json \
  --expected-candidate "$(git rev-parse HEAD)" --require-samples 20 --enforce-gate
```

This is the supported performance envelope, not a general latency promise. The
evidence covers one native CPU extension, immutable packed batches of 16, the
fixed fragmented/restorative workload, and the recorded CI environment. It does
not qualify larger live states, other batch sizes, arbitrary mutation shapes,
concurrent load, or end-to-end application latency. Sorted-vector staging is
linear in live interval count; the matrix intentionally exposes that scaling
cliff rather than extrapolating the faster 64-interval result.
