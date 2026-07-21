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
python -m tests.performance.exact_batch_benchmark \
  --output build/benchmarks/exact-batch.json
```

When an output is requested, the benchmark atomically publishes canonical JSON,
a concise Markdown summary, and a SHA-256 sidecar. The JSON binds the exact
candidate commit and clean-worktree state, runtime/compiler/build and native
binary metadata, the independently reconstructable restorative workload manifest
and digest, raw paired samples, bootstrap methodology, fixed thresholds, and all
locally callable gate derivations. Standard output remains the complete diagnostic
JSON. `scripts/verify_exact_batch_benchmark.py` rejects duplicate keys and
recomputes the workload, every interval, speedup, throughput bound, and gate from
raw samples; it also binds Markdown and the checksum to the canonical JSON bytes.

`--enforce-hard-gates` is the local batch-only gate. It enforces, without lowering
thresholds:

- batch-16 throughput lower 95% bound at least 2 million logical operations/s;
- batch-16 speedup lower 95% bound at least 2x stable scalar;
- break-even lower 95% bound by batch size 4.

Complete hosted promotion evidence additionally requires a separate, clean
baseline/candidate `mutation_attribution.py --quick` artifact. The exact-batch
verifier structurally verifies that artifact, independently recomputes the paired
`rangeset_public` bootstrap interval, binds its candidate commit to the exact-batch
artifact, and requires its upper 95% candidate/baseline bound to be at most 1.03.
This bounded quick/layers comparison supports only the stable scalar regression
gate. It omits representative and control workloads and must not be described as
full scalar-promotion evidence. The fixed 1.03 threshold is not relaxed when a
local or hosted run is inconclusive.

The path-filtered `Experimental exact-batch evidence` pull-request/manual workflow
builds the C++ CPU extensions, creates a clean baseline worktree, runs both paired
measurements with 30 samples, verifies all six artifact files, enforces all four
gates, retains the evidence for 90 days, and publishes the bound commits, digests,
and gate values in the Actions summary. A failed or inconclusive scalar upper
bound leaves promotion evidence incomplete.

Sorted-vector staging is intentional, not a generic abstraction. Its cost scales
with live interval count on every batch and therefore has a vector-scaling cliff:
small batches or substantially larger fragmented states can erase the benefit.
Promotion requires evidence on the stated 64-interval workload and explicit
characterization of that cliff; this experiment is not a claim that vector state
is universally faster.
