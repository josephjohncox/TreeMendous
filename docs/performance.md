# Performance

Tree-Mendous performance is a property of an interface, backend, state shape,
operation mix, and machine. The measurements below are scoped evidence, not
universal speed claims or promises for an unmeasured deployment.

## Current local standard measurements

The current local `standard` profile was measured at commit
`ec91793d0dbd5152b3bb2baf4231297df92692f0` on an Apple M5 Max running macOS
26.5.1 and CPython 3.12.7. It used 20 independent runs per backend and workload,
two warmups, and six deterministic traces starting with 64–128 intervals. Every
accepted run was replayed and checked against the independent oracle.

For `cpp_boundary`, aggregate timed-public-call throughput ranged from **0.30M
to 0.91M public operations per second** across the six traces. Depending on the
trace, this was 1.11×–3.15× the throughput of the fastest stable pure-Python
backend in the same run. Selected median public-call observations were:

| Operation | Observed median range | Scope |
| --- | ---: | --- |
| `add` / `discard` | about 1.21–1.33 µs | geometry mutations in the sampled traces |
| `first_fit` | about 1.54 µs | bounded fit queries in traces that issue them |
| `allocate` | about 0.69–2.08 µs | atomic fit-and-remove calls; state shape differs by trace |
| `snapshot` | 2.76 µs to 21.6 µs in the compact traces | 2.76 µs at 64 intervals; 21.6 µs in the 128-interval catalog |

The separate lease trace recorded a **585 µs** snapshot. That outlier is
reported separately because it is workload- and state-shape-dependent; it
demonstrates that snapshot cost cannot be inferred from an initial interval
count alone.

### What these timings include

The harness constructs arguments and resolves methods before starting each
public-call timer. The timer then covers the Python bound-method call through
its return or expected exception. For `RangeSet`, that includes public wrapper
validation performed inside the method, backend dispatch, geometry work, and
construction of the returned public object. For `snapshot`, it includes
constructing the returned snapshot and interval values.

The public-call timer excludes setup, warmups, construction of `Span` arguments,
harness bookkeeping, post-return result freezing and canonicalization,
checksums, and independent-oracle validation. Aggregate throughput divides the
number of declared public calls by the sum of those call durations; it is not
end-to-end request throughput. Operation medians describe calls within fixed
traces and are not independent deployment latency samples.

The complete checked evidence is attached to the 1.1.0 release as
[JSON](https://github.com/josephjohncox/TreeMendous/releases/download/v1.1.0/treemendous-rangeset-standard-ec91793.json),
[Markdown](https://github.com/josephjohncox/TreeMendous/releases/download/v1.1.0/treemendous-rangeset-standard-ec91793.md),
and a
[SHA-256 sidecar](https://github.com/josephjohncox/TreeMendous/releases/download/v1.1.0/treemendous-rangeset-standard-ec91793.json.sha256).
The JSON digest is
`sha256:af70331fad467ba3cf6eeb3f24b2733b15e05dee4ee4e1acc2f7053cdfdb78e1`.
Run the same correctness-checked profile on the target machine with:

```bash
just benchmark-standard
```

The [benchmark methodology](benchmarking.md) documents workload generation,
validation, confidence intervals, and durable JSON/Markdown/checksum artifacts.

## Hosted 1.1.0 exact-batch evidence

The hosted 1.1.0 release evidence was produced by GitHub Actions
[run 29879842390](https://github.com/josephjohncox/TreeMendous/actions/runs/29879842390).
The original Actions
[artifact 8514386947](https://github.com/josephjohncox/TreeMendous/actions/runs/29879842390/artifacts/8514386947)
has digest
`sha256:7b63e752b0f12c765c0e099e7229d4f3492da7fd364cc370dda0d4b9860732d9`.
Durable release assets preserve the verified
[exact-batch JSON](https://github.com/josephjohncox/TreeMendous/releases/download/v1.1.0/treemendous-exact-batch-1.1.0.json),
[scaling JSON](https://github.com/josephjohncox/TreeMendous/releases/download/v1.1.0/treemendous-exact-batch-scaling-1.1.0.json),
and
[scalar-attribution JSON](https://github.com/josephjohncox/TreeMendous/releases/download/v1.1.0/treemendous-scalar-attribution-1.1.0.json)
with their Markdown reports and SHA-256 sidecars. The release gates used
conservative confidence bounds rather than point estimates:

| Evidence | Hosted result |
| --- | ---: |
| batch-16 lower-bound throughput | 3,533,569 logical operations/s |
| batch-16 lower-bound speedup | 8.49× |
| batch-4 lower-bound speedup | 4.59× |
| stable scalar candidate/baseline ratio, 95% CI | 0.9999–1.0133 |
| 100,000-live-interval batch-16 latency, upper 95% bound | 1.541 ms |
| process peak RSS | 149.1 MiB |

Exact-batch throughput counts logical mutation rows. Its timed layer is one
`mutate_packed` call, including native staging, ordered execution, atomic
publication, and packed-result construction. It excludes domain and manager
setup, canonical replay, materialization of Python `MutationResult` objects,
snapshots, and correctness validation. The scalar ratio uses paired,
correctness-checked stable `cpp_boundary` traces and is evidence for those fixed
traces, compiler settings, and hosted machine only. Peak RSS is process-wide,
not the retained size of one range-set object.

The 100,000-interval row qualifies one fixed sorted-vector workload. It does not
show constant-time scaling, arbitrary batch latency, application-engine
latency, or a general memory ceiling. See the
[1.1.0 release notes](releases/1.1.0.md) and
[exact-batch contract](exact-batch.md) for the stable semantic envelope.

## Locking levels and the fully-native scalar path

`RangeSet` exposes two first-class, comparable ways to reach the same
authoritative `cpp_boundary` geometry, plus a selectable locking level. Both are
reported alongside the plain-native floor by
[`tests/performance/rangeset_hotpath_benchmark.py`](../tests/performance/rangeset_hotpath_benchmark.py),
verified by [`scripts/verify_rangeset_hotpath_benchmark.py`](../scripts/verify_rangeset_hotpath_benchmark.py),
and generated with `just benchmark-hotpath`.

### Mutation interfaces

- `add(span)` / `discard(span, require_covered=...)` return a `MutationResult`
  describing the exact changed geometry, changed length, and pre-mutation
  coverage. This is the default, evidence-rich surface.
- `release(span)` / `reserve(span, *, require_covered=...)` are the fully-native
  scalar mutators. They apply the identical geometry mutation and return only
  the integer `changed_length`, building no `Span`/`MutationResult`. They are
  exactly geometry-consistent with `add`/`discard`. They require an
  authoritative geometry backend and no payload policy; otherwise they raise
  `ValueError` rather than falling back silently. `require_covered` rejects a
  span that is not fully covered as a `0` no-op.

### Locking level (`synchronized`)

`RangeSet(..., synchronized=True)` (the default), `create_range_set(...,
synchronized=True)`, and `BackendRegistry.create(..., synchronized=True)` keep
the reentrant internal lock and the existing concurrent snapshot/mutation
consistency. `synchronized=False` installs a no-op lock: the reentrancy guard
still rejects nested mutations, but **there is no internal synchronization**.
The caller alone guarantees single-threaded access or supplies external
locking, and snapshot/mutation consistency becomes the caller's
responsibility. Choose `synchronized=False` only when the caller already owns a
coarser lock or the range set is confined to one thread.

### Scoped measurements

Measured on an Apple M5 Max (macOS 26.5.1, CPython 3.12.7) over a deterministic
restorative trace on a single validated `cpp_boundary` instance, 30 samples,
timing only the public mutation calls (construction, setup, and correctness
validation excluded):

| Path | Median throughput | Scope |
| --- | ---: | --- |
| `add`/`discard` (MutationResult), synchronized | about 1.54M ops/s | default locked geometry mutations |
| `add`/`discard` (MutationResult), unsynchronized | about 1.65M ops/s | no internal lock; caller owns synchronization |
| `release`/`reserve` (scalar), synchronized | about 4.32M ops/s | native changed-length only, locked |
| `release`/`reserve` (scalar), unsynchronized | about 5.06M ops/s | native changed-length only, caller-synchronized |
| plain-native floor | about 7.9M ops/s | raw `IntervalManager` scalar mutators, no wrapper |

This is scoped evidence for one interface family, one restorative workload, one
host, and one timed layer. It is not a universal speed claim. Two levers
compound. Dropping eager `MutationResult`/`Span` construction — the scalar
`release`/`reserve` path returns only the changed length — is the larger win,
roughly tripling throughput over `add`/`discard` while staying exactly
geometry-consistent. Choosing `synchronized=False` then removes the per-op
`with self._lock:` protocol from the hot path (the context-manager
enter/exit calls, not lock contention, are the cost), lifting the scalar path
from about 4.3M to about 5.06M ops/s — within roughly 65% of the raw-native
floor. Combined, the unsynchronized scalar surface is about 5.5x the default
`add`/`discard` throughput. The unsynchronized level performs no internal
locking: the caller alone owns cross-thread synchronization and
snapshot/mutation consistency.

## Choosing the faster appropriate interface

- Prefer `release`/`reserve` over `add`/`discard` when the changed length is the
  only result you need on an authoritative geometry-only range set.
- Use `synchronized=False` only under a caller-owned locking discipline or
  confirmed single-threaded use; keep the default otherwise.
- Use `RangeSet` when payload policies, allocation, generic queries, backend
  selection, or protocol compatibility are required.
- Use `ExactBatchRangeSet` now for ordered, whole-batch-atomic,
  **geometry-only** mutations. Do not discard payload or application semantics
  merely to reach the native batch path.
- Measure the concrete application operation when using one of the 50 engines.
  Generic geometry throughput does not include its identity, lifecycle,
  scheduling, or fencing logic.
- Treat every multidimensional index as experimental and qualify it with the
  target distribution of boxes and queries.

## Optimization roadmap

This is a staged roadmap, not functionality delivered by 1.1.1:

1. **Use `ExactBatchRangeSet` now for geometry-only batches.** Keep resource
   limits explicit and retain external domain semantics outside the geometry
   object.
2. **Add an explicit `RangeSet` transaction or mutate-many path.** It must
   preserve current payload-policy algebra, ordered observations, validation,
   mutation accounting, and atomic rollback semantics rather than routing
   payload-bearing state through the geometry-only exact-batch API.
3. **Replace full-state copying at large N.** Segmented or persistent native
   storage should make mutation publication proportional to touched segments
   instead of copying an O(N) sorted vector for every batch.
4. **Make snapshots lazy or structurally shared.** Preserve immutable,
   point-in-time semantics while avoiding eager recreation of every interval
   when callers need only totals or a short-lived view.
5. **Inject backends into concrete engines only after qualification.** Define
   each engine's required capabilities, payload/identity boundary, rollback
   behavior, and benchmark gate before allowing backend choice.
6. **Change representation only behind fixed gates.** Semantic parity,
   adversarial rollback, concurrency, memory, scalar latency, batch
   break-even, large-state latency, and snapshot-cost gates must remain fixed
   while candidate structures are compared.

No speculative runtime, storage, snapshot, or application-integration change is
part of the 1.1.1 documentation and metadata patch.

## Experimental concrete-application backend qualification

The Phase D experiment routes the private factories used by the contiguous and
disk allocators, scoped lease pools, claim ledger, and partition runtime through
the semantically probed backend registry. It exercises construction and every
checkpoint/rebuild/restore path without changing public constructors or stable
runtime selection. Run the bounded matrix and then independently verify its
canonical JSON, Markdown, checksum, derivations, and provenance:

```bash
just experiment-application-backend-matrix
just verify-application-backend-matrix
```

The command evaluates every locally available stable deterministic signed-64
CORE backend against `py_boundary`, uses 20 balanced paired blocks per cell,
and records fail-closed evidence for ineligible requests. A `REJECTED` result is
expected evidence, not a command failure; no application backend injection is
retained unless all fixed initial and confirmation gates pass.
