# Benchmarking and bounded load evidence

Tree-Mendous has two application-related benchmark suites with different jobs.
The generic suite qualifies `RangeSet` behavior and stable backends. The
concrete suite executes the 50 registered application engines. A result from one
suite is not evidence for the other.

## Commands

Generic geometry and backend profiles:

```bash
just benchmark-smoke
just benchmark-standard
just benchmark-large
```

Concrete application profiles:

```bash
just benchmark-applications-smoke
just benchmark-applications-standard
```

`just benchmark` runs both standard suites. `just run-examples` executes the 50
application examples plus the basic, exact-batch, and multidimensional examples
from an unrelated working directory.

A manually dispatched GitHub Actions run publishes verified generic benchmark
artifacts for 90 days:

```bash
gh workflow run benchmarks-adhoc.yml \
  --ref <pushed-branch-or-sha> \
  -f profile=standard \
  -f section=sampled
```

The default sampled run includes both `canonical-local-mutation-throughput` and
`observed-fragmented-mutations`. Before upload, the workflow independently
checks the JSON digest, schema, profile, section, commit, stable backend set,
required workloads, and Markdown digest reference. The artifact contains the
canonical JSON, Markdown, and SHA-256 sidecar and is linked from the Actions job
summary. GitHub-hosted runner timings remain directional rather than a stable
cross-run performance baseline.

## Performance layers are not interchangeable

Tree-Mendous has three distinct timing boundaries:

1. A **raw geometry kernel** calls an internal Python or C++ implementation
   directly. This isolates its data structure and language boundary but omits
   the stable API contract.
2. A **canonical backend benchmark** calls `RangeSet`. It includes domain
   validation, locking, exact mutation evidence, canonical result objects,
   cache publication, and any payload work.
3. A **concrete application benchmark** times one domain action. That action can
   include identity checks, idempotency, event records, retries, normalization,
   and several `RangeSet` calls.

A raw C++ mutation rate is not a `RangeSet` throughput claim, and a `RangeSet`
operation rate is not an application-engine throughput claim. Reports must name
the layer. Comparisons are valid only when the workload, operation definition,
state cardinality, fragmentation, validation boundary, and sample method match.

Every command writes canonical JSON, a Markdown summary, and a SHA-256 sidecar
under `build/benchmarks/`. The default smoke outputs are:

```text
build/benchmarks/smoke.json
build/benchmarks/smoke.md
build/benchmarks/smoke.json.sha256
build/benchmarks/applications-smoke.json
build/benchmarks/applications-smoke.md
build/benchmarks/applications-smoke.json.sha256
```

Pass an output path as the recipe argument to retain several runs:

```bash
just benchmark-applications-smoke \
  build/benchmarks/applications-smoke-$(git rev-parse --short HEAD).json
```

## Generic backend trace suite

`tests.performance.benchmark_suite` exercises the public `RangeSet` API. The
`just benchmark-smoke`, `just benchmark-standard`, and `just benchmark-large`
recipes build the native CPU extension and require all six stable CPU backends.
The suite covers six core workload shapes:

- exact local reserve/release mutation throughput over fragmented geometry;
- fragmented allocator churn with reserve, release, lookup, allocation, and
  impossible-fit cases;
- fragmented local mutations followed by exact canonical snapshots, so lazy
  publication costs are measured rather than hidden in write-only traces;
- immutable read-heavy catalogs with fit, overlap, snapshot, and statistics
  checkpoints;
- bounded scheduling traces with release coordinates, deadlines, cancellation,
  occupancy pressure, success counts, and Jain fairness;
- sharded numeric lease-pool traces with block allocation and stale or duplicate
  cancellation.

A payload section separately replays uniform, join, and ordered payload policies
through every stable geometry backend.

The generic suite also has an `applications` section containing 50 legacy
application-labeled traces. Each label maps to one of the generic workload
constructors above. For example, `distributed-document-search` runs a generic
lease-pool-style range trace; it does not construct or search a document index.
These traces compare backend range semantics and do not contribute application
completion evidence. The [legacy workload matrix](use-cases.md#legacy-generic-backend-qualification-traces)
records each label and its range interpretation.

### Generic profiles

| Profile | Purpose | Sampled scale | Bounded load observations | Legacy traces |
| --- | --- | --- | --- | ---: |
| `smoke` | Required PR/build semantic check | 32 to 128 initial ranges and 200 to 1,100 operations per workload, 20 samples | 500-range catalog and 250-shard lease pool | 40 operations for each of 50 labels |
| `standard` | Weekly engineering run | Up to 128 initial ranges and 1,100 operations, 20 samples | 10,000-range catalog, 2,000-shard lease pool, and 25,000 scheduled jobs | 100 operations for each label |
| `large` | Manual high-cardinality observation | Up to 128 initial ranges and 1,100 operations, 20 samples | 25,000-range catalog, 5,000-shard lease pool, and 50,000 scheduled jobs | 200 operations for each label |

The large profile is a bounded synthetic observation. It does not define
production capacity or a supported deployment envelope. The profile separates
interval cardinality from operation count where combining both would produce an
unhelpful cross product for a linear-scan backend.

For generic geometry workloads, the harness uses an independent sorted-list
oracle with its own span type, validation, index, mutation implementation, and
accounting. Before a timing sample is accepted, replay must match all fit,
allocation, overlap, snapshot, statistics, mutation, and final-state evidence.
Normalization, checksums, and divergence rejection occur outside the timed
replay.

The sampled generic measurements use at least 20 independent runs, warmups,
deterministically randomized backend order, medians, median absolute deviation,
and a run-level bootstrap interval for the median. Schema v4 persists the raw
run-level setup, execution, validation, and per-operation run-median timing
samples as well as their robust summaries. The single timed replay used for
each large qualification is only a load observation. No generic profile enforces a wall-clock regression
threshold.

## Paired native-mutation attribution

`tests.performance.mutation_attribution` is a separate paired diagnostic. It
force-builds both source roots before sampling, hashes the imported native
extensions, records compiler output and build flags, and alternates
baseline/candidate process order for each of at least 20 paired rounds. Every
cumulative boundary uses the standard 64-interval, 1,000-mutation, seed-50
trace. Setup, callable resolution, Python argument-object construction,
evidence freezing, checksums, and final-state validation are outside timing for
every boundary. Checked scalar conversion and all work performed by the invoked
production callable remain inside timing.

The cumulative boundaries are:

- `binding_no_result`: `release_interval` / `reserve_interval`, without exact
  result construction;
- `binding_result`: production `release_with_delta` / `reserve_with_delta`,
  including the native preview, Python `Span` and `MutationResult` construction,
  and the subsequent native mutation;
- `adapter`: authoritative `BackendAdapter` dispatch and result type enforcement;
- `rangeset_public`: production `RangeSet.add` / `discard`, including public
  validation, locking, guarding, accounting, and cache invalidation;
- `observed_publication`: public mutation immediately followed by `snapshot()`.

Differences between cumulative boundaries are diagnostic; policy evaluation
uses directly timed boundaries rather than subtracting noisy samples. An
untimed replay of every boundary must reproduce the same per-operation changed
spans, `changed_length`, `fully_covered`, counters, and final-state checksum.
A full comparison also pairs every exact standard representative workload for
`cpp_boundary`, the five Python environmental controls, and four focused traces:
no-op mutations, allocation hits, fragmented unbounded allocation misses, and
bounded allocation misses.

For paired round `i`, the comparator records
`r_i = candidate_ns_i / baseline_ns_i`, takes `median(r_i)`, and bootstraps the
paired ratios with 10,000 fixed-seed resamples. It persists raw paired samples,
paired ratios, median improvement, the versioned exact representative-workload
manifest and digest, trace digests, and each round's timing, semantic,
environment, and binary-provenance evidence. The verifier reconstructs the
standard manifest independently and recomputes those derived fields.

There is no repository-wide attribution threshold. Diagnostic generation takes
no policy by default and exits successfully after writing a semantically valid
artifact even when an explicitly supplied policy classifies performance as
`fail` or `inconclusive`. A full, clean, environment-matched run can become
promotion-eligible only when all policy inputs were explicitly supplied. Dirty
and `--quick` runs remain diagnostic. The explicit gate command additionally
requires the caller's expected limits and rejects dirty, quick, failed, or
inconclusive artifacts; there is no dirty bypass.

```bash
# Diagnostic generation and structural/derived-field verification.
just benchmark-attribution /absolute/path/to/baseline \
  /absolute/path/to/candidate \
  build/benchmarks/mutation-attribution.json
just verify-attribution build/benchmarks/mutation-attribution.json

# To produce a policy-bearing artifact, pass all four policy bounds explicitly.
uv run python -m tests.performance.mutation_attribution \
  --baseline-root /absolute/path/to/baseline \
  --candidate-root /absolute/path/to/candidate \
  --samples 20 --warmups 1 \
  --primary-ratio-limit "$PRIMARY_LIMIT" \
  --regression-ratio-limit "$REGRESSION_LIMIT" \
  --control-ratio-minimum "$CONTROL_MIN" \
  --control-ratio-maximum "$CONTROL_MAX" \
  --output build/benchmarks/mutation-attribution.json

just gate-attribution build/benchmarks/mutation-attribution.json \
  "$PRIMARY_LIMIT" "$REGRESSION_LIMIT" "$CONTROL_MIN" "$CONTROL_MAX" 20
```

The comparator writes canonical JSON, Markdown, and a SHA-256 sidecar. It does
not publish an operations-per-second headline: throughput is derived only from
measured elapsed time and the declared operation count in the artifact. Native
attribution is not a mandatory scheduled parent-commit promotion gate.

## Segmented ExactBatch storage qualification

The Phase G2 storage experiment compared the immutable vector baseline at
commit `2a384f7` with the now-rejected segmented candidate in isolated
subprocesses. The durable proof of the rejection is the tracked patch
`tests/performance/experiments/fixtures/exact_batch_segmented_tuned.patch` plus
its pinned SHAs: applying it to the baseline `exact_batch_bindings.cpp` at
`2a384f7` reproduces the segmented source SHA-256
`f5a368f011bcbbe9f49ba954b7014268ab7c711ecfb1d46b8b4d9da6a8858267`. That patch
chain is checked unconditionally by the storage experiment test suite and does
not depend on any measured artifact.

The measured 20-block tuned-smoke triplet is reproducible evidence, not tracked
source: like every other benchmark artifact it lives under the ignored
`build/`/`docs/evidence/` output tree or is attached as a release asset. Once a
copy exists at
`docs/evidence/experiments/exact-batch-storage-segmented-tuned-rejection.json`
(with its Markdown and `.json.sha256` sidecars), verify it offline without the
historical binaries:

```bash
just verify-exact-batch-storage-archive
```

The archive binds baseline commit/source/binary hashes, candidate binary and
runtime provenance, the exact reconstructed patch SHA-256, and the resulting
segmented `exact_batch_bindings.cpp` SHA-256.
Verification reapplies the patch to the baseline source from Git and strictly
reconstructs worker schemas, all scalar-oracle digests, raw block ratios,
bootstrap summaries, matrix order, and the rejected 1.10 small-N gate. It does
not require the rolled-back package to expose segmented counters. The
archive-data contracts in the test suite skip automatically when no such copy is
present.

To reproduce rather than verify, use the intentionally expensive recipe below.
It creates two detached temporary worktrees at the baseline commit, applies and
hash-checks the patch only in the candidate, builds distinct binaries, invokes
the fixed smoke harness for 20 balanced blocks, and removes both worktrees on
exit:

```bash
just reproduce-exact-batch-storage-rejection \
  build/experiments/exact-batch-storage-segmented-reproduced.json
```

The fixed matrix covers N={64,1000,10000,100000}, B={0,1,16,256}, local,
strict-only, duplicate-only, 1/10/100%-width, and K-1/K/K+1 block-boundary
cells. Twenty fixed blocks alternate which isolated root process runs first.
Each process measures the complete promotion matrix; bootstrap intervals
resample whole block ratios. Setup, canonical `cpp_boundary` replay, packed
result materialization, final snapshots, counter inspection, RSS, and artifact
writing are outside mutation timers. Construction, mutation, snapshot, and
materialization are separate cells.

The verifier requires canonical JSON plus matching Markdown and SHA-256,
rejects duplicate keys and non-finite values, reconstructs matrix identity and
every ratio/bootstrap/gate derivation, and recomputes source, runtime, compiler,
and native-binary provenance. The pre-registered gates are not adjusted after
measurement. A failed required gate rejects the segmented runtime rather than
weakening a threshold. The sole small-N tuning confirmation was rejected: its
20-block N64/B16 candidate/vector median was 1.065273 and its upper-95 bound was
1.115916, above the unchanged 1.10 gate. The package therefore retains vector
storage and does not expose the segmented counters or failpoints.

## Experimental exact-batch application matrix

The first roadmap experiment is a correctness-attested diagnostic under
`tests/performance/experiments`; it is not stable package code and does not
change the existing exact-batch gates. A bounded smoke run is:

```bash
just experiment-exact-batch-application-matrix smoke \
  build/experiments/exact-batch-application-smoke.json
just verify-exact-batch-application-matrix \
  build/experiments/exact-batch-application-smoke.json
```

The complete local profile uses at least 10 paired raw samples for every cell
across batch sizes 1, 4, 16, and 64; pre-call live state counts 64, 1,000, and
10,000; head/middle/tail locality; and strict accept/reject, idempotent
real/no-op, fragment/restore, coalesce/restore, and eight-span fan-out shapes:

```bash
just experiment-exact-batch-application-matrix local \
  build/experiments/exact-batch-application-local.json
```

The 100,000-state slice is manual and explicitly opt-in:

```bash
uv run python -m tests.performance.experiments.exact_batch_application_matrix \
  --profile local --include-100000 --samples 10 \
  --output build/experiments/exact-batch-application-100000.json
```

Manager/domain setup, operation packing, scalar-oracle construction, result
materialization, snapshots, validation, and artifact writing are outside both
timers. Each paired sample alternates packed-first/scalar-first order. Every
packed row and final state is checked against scalar `cpp_boundary` replay.
The JSON records raw packed/scalar samples, paired ratios, bootstrap median
intervals, packed input/result bytes, exact declared work, workload digest, and
source/build/environment provenance; Markdown and SHA-256 companions bind the
report.

Interpret observed median break-even only for the named state, shape, locality,
machine, build, and timed layers. A confidence interval crossing 1.0 is
inconclusive. A local dirty-worktree artifact remains useful for diagnosis but
is not promotion evidence. This experiment makes no universal performance
claim and leaves the stable v3/v1 gates and their workflows unchanged.

## Experimental payload-aware RangeSet transaction

The private `tests.performance.experiments.rangeset_transaction` prototype
copies complete geometry, payload, and ordered-event state to a fresh proven
catalog backend, replays ordered scalar mutations, and publishes only after all
fallible work succeeds. It adds no `RangeSet` method or stable protocol. Run the
short diagnostic or the complete B={0,1,4,16,64}, N={64,1000}, four-policy,
two-trace matrix with:

```bash
just experiment-rangeset-transaction bounded \
  build/experiments/rangeset-transaction-bounded.json
just experiment-rangeset-transaction full \
  build/experiments/rangeset-transaction-full.json
just verify-rangeset-transaction \
  build/experiments/rangeset-transaction-full.json
```

Every cell alternates transaction/scalar order for at least 15 paired samples;
setup and exact same-instance result/final-state validation remain outside the
timers. `tracemalloc` is inactive for both latency paths; peak memory comes from
a separate untimed transaction replay that is also validated against the scalar
oracle. The canonical JSON/Markdown/SHA-256 triplet records raw total, staging,
backend-load, memory, interval, resource, source, and backend provenance. Fixed
gates require B16/N1000 upper-95 ratios at most 1.00 for every payload, one
non-restorative upper-95 ratio at most 0.90, and no nonempty cell above 1.10.
Failure marks the experiment `REJECTED`; it is not a reason to weaken a gate or
promote the candidate into the stable package.

## Experimental geometry snapshot cache

The stable `RangeSet` reuses one exact immutable `RangeSnapshot` only for an
unchanged geometry-only state. Payload-bearing snapshots retain their existing
clone-and-detach behavior. The E4-A scaling experiment compares public cached
`snapshot()` with faithful uncached `RangeSnapshot` construction on the same
state. Setup and same-instance validation remain outside retained-block timing:

```bash
just experiment-rangeset-snapshot-scaling \
  build/experiments/rangeset-snapshot-scaling.json
just verify-rangeset-snapshot-scaling \
  build/experiments/rangeset-snapshot-scaling.json
```

The pre-registered N={100,1000,10000} confirmation matrix uses exactly 40
balanced blocks, not sample-until-pass collection. Every retained block contains
both a cached-first and an uncached-first ordering; the ordering-pair sequence is
reversed in alternating blocks. Cached elapsed time is totaled across its two
positions, uncached elapsed time is totaled across its two positions, and one
ratio is then derived for the whole block. Bootstrap 95% median intervals
resample whole blocks, never pooled individual calls. Unchanged 16-read bursts
use the same balanced-block analysis as restorative write-then-observe cycles.

Each timed position repeats one common number of cached or uncached iterations.
An excluded pilot starts at two iterations and doubles until both position
durations reach at least 5 ms or the deterministic 64-iteration cap. The JSON
records every pilot duration, the selected count, every raw position and block
total, and the derivations. Pilot observations never enter samples, bootstrap
intervals, or gates. Fixed acceptance gates remain unchanged: the N=10000
unchanged upper-95 cached/uncached ratio must be at most 0.25, the cached
N=10000/N=1000 per-iteration median ratio at most 1.50, and every
write-then-observe upper-95 ratio at most 1.10. Canonical JSON, Markdown, and
SHA-256 artifacts bind exact-type verification plus source, worktree, runtime,
and backend-binary provenance.

This balanced-block protocol corrects an order-confounding defect in the prior
pooled single-call analysis. The earlier artifact treated AB and BA calls as
exchangeable despite a strong order effect and is invalidated; it is not
confirmation evidence for accepting or rejecting the cache.

## Lease-state publication scaling

The E4-B/D lease-state experiment writes a canonical JSON/Markdown/SHA-256
triplet and supports generation and verification as distinct commands:

```bash
just experiment-lease-state-scaling 30 \
  build/experiments/lease-state-scaling.json
just verify-lease-state-scaling \
  build/experiments/lease-state-scaling.json
```

The fixed matrix is the ordered Cartesian product of four workload kinds and
N={128,512,2048,8192}. The verifier requires exact per-kind schemas and JSON
types, recomputes every balanced block total, ratio, bootstrap summary, gate,
semantic result/final-state digest, fixed instrumentation count, and retained
memory settling ratio, and binds current source/runtime provenance. Generation
performs semantic validation and digest construction outside timing. The
verifier rejects noncanonical bytes, duplicate keys or matrix rows, relabeling,
missing fields, and numeric bool/int coercion.

## Experimental radio-spectrum index representation

The E6 experiment injects `BoxIndex2D` axis projection and a guarded
`BoundedBoxIndex` into an empty, real `RadioSpectrumScheduler` only from the
performance harness. The stable scheduler constructor remains unchanged and
continues to construct linear `BoxIndex(2)`. Run the fixed matrix with an
explicit one-hour subprocess timeout, then independently verify its artifact:

```bash
just experiment-radio-spectrum-index-matrix \
  build/experiments/radio-spectrum-index-matrix.json 3600
just verify-radio-spectrum-index-matrix \
  build/experiments/radio-spectrum-index-matrix.json
```

The training sizes are 32, 128, 512, 2,000, and 10,000 active entries. Separate
deterministic held-out sizes are 64, 256, 1,000, and 5,000. Five workload cells
cover materially different low, medium, and high-overlap seed/query geometry;
narrow and broad channel/time queries; 3:1, 2:2, and 1:3
insertion/cancellation call mixes; and balanced, channel, time, and dual-axis
skew. The fifth, separately named `idempotent-replay` cell measures replay and
is not counted as any insertion/cancellation mix. Timed mix insertions use
distinct request IDs and reservations, and cancellations target distinct live
handles; any live-handle preparation is untimed and independently replayed.
Every timed position contains eight real scheduler operations. Twenty-five
fixed AB/BA blocks are recorded before the run, and bootstrap intervals
resample whole blocks. Construction, seeding, preparation, independent-oracle
replay, exact same-instance validation, snapshots, diagnostics,
candidate/posting observations, retained-graph memory, and the RSS high-water
proxy remain outside operation timers.

The sparse grid is constructed only with explicit finite channel/time bounds,
cell sizes, and all resource guards. A broad-query adversary deliberately
exceeds `max_cells_per_query`; its `ValueError` must propagate with unchanged
state and no catch-and-fallback behavior. Correctness evidence also covers
reservation and conflict results, duplicate request identity and idempotency,
cancellation, snapshot order/version, diagnostic algorithm, and exact final
state against an independent sorted-list radio oracle and the linear scheduler.

Fixed gates are not tuned by a run. A representation needs upper-95 latency
ratio at most 0.80 at two adjacent training sizes; its crossover is the lower
size in that passing adjacent pair. Every selected crossover must have at least
one predetermined held-out size at or above the crossover, and empty held-out
evidence fails qualification. A policy may select only training cells at most
0.90; every applicable selected held-out cell must also be at most 0.90. A
balanced raw-block control separately measures the untouched default scheduler
against an explicitly injected linear `BoxIndex(2)` and requires an upper-95
ratio at most 1.10. Selected retained memory must be at most 1.25 times linear.
There is no live migration. Failure to pass every gate retains no scheduler
factory seam and leaves runtime linear. The prior full artifact predates this
v2 protocol and is marked stale; it is not qualification evidence. The strict
canonical JSON/Markdown/SHA-256 verifier reconstructs exact Cartesian
membership and order and rejects row relabeling or duplication, duplicate keys,
non-finite numbers, exact-type changes, query-diagnostic derivation or final
state digest tampering, decision tampering, and source/runtime/backend
provenance drift.

## Exact-batch evidence

The stable specialized exact whole-batch module has a separate path-filtered
pull-request and manual workflow. Its stable v3 batch-local artifact is not
interchangeable with the stable native-mutation attribution report. The
exact-batch producer writes
canonical JSON, Markdown, and a SHA-256 sidecar containing raw paired samples,
the exact candidate commit and clean state, compiler/build/native-binary metadata,
and a versioned restorative workload manifest. One packed result from every
timed sample is retained and materialized after timing; its exact per-row values
and both timed instances' final states are checked against the canonical scalar
oracle. Destruction for the other packed results remains timed, while the one
retained result and all validation are excluded. Materialization has a separate
reported timing. The strict verifier reconstructs the workload, compares JSON
types exactly, and recomputes paired ratios, bootstrap intervals, throughput,
speedup intervals, and fixed gates rather than trusting stored booleans.

A local batch gate is explicitly callable with:

```bash
uv run python -m tests.performance.exact_batch_benchmark \
  --samples 20 \
  --output build/benchmarks/exact-batch.json \
  --enforce-hard-gates
```

That command covers only batch-4 break-even and batch-16 throughput/speedup. The
hosted promotion lane additionally checks the fixed stable scalar regression gate
against clean baseline and candidate commits. For bounded runtime it uses the
existing mutation-attribution producer in `--quick` mode and independently
recomputes the `rangeset_public` paired bootstrap interval. The upper 95%
candidate/baseline bound must be no greater than 1.03. Quick mode intentionally
omits representative workloads and Python controls, so this artifact is scoped
only to stable scalar regression and is not full scalar-promotion evidence. An
inconclusive interval fails the complete exact-batch promotion check; the 1.03
limit is never widened.

A separate scaling producer qualifies batch-16 calls over 64, 1,000, 10,000,
and 100,000 disjoint live managed-domain components. Setup is excluded from
timing. Deterministic remove/restore, strict-reject, and no-op rows target the
end of the sorted vector, and every call begins and ends with N intervals. Every
result row and final snapshot is checked against stable canonical semantics
after timing. Each matrix case stores at least 20 raw batch-latency samples, a
fixed-seed bootstrap median interval, logical throughput, packed-result bytes,
process peak RSS, exact environment/commit/native-binary provenance, workload
digests, and the production `BatchLimits` values.

The fixed envelope gate requires the upper 95% median-latency bound for 100,000
intervals and batch size 16 to be no greater than 10 ms. This bounded evidence
makes the sorted-vector scaling cliff visible: staging work grows linearly with
live interval count. It does not support extrapolation past 100,000 intervals or
to different batch sizes, workloads, concurrency, machines, or application
latency. The original 64-interval 2,000,000 operations/second, 2x batch-16,
batch-4 break-even, and scalar-regression gates remain unchanged.

The workflow uploads three verified JSON/Markdown/checksum triplets for 90 days:
exact-batch, scoped scalar attribution, and scaling. All three bind the same
candidate SHA. The Actions summary records candidate/baseline commits, workload
and JSON digests, and fixed gate values. GitHub-hosted timing remains evidence
for this bounded lane and workload, not a general machine-independent
performance claim.

## Concrete application suite

`tests.performance.application_benchmark_suite` is separate. It iterates the
canonical `SCENARIO_SPECS`, imports the exact benchmark module registered for
each complete scenario, and calls its `run_benchmark` function. Each module
constructs and executes the named engine, then compares that same timed engine
instance with the scenario's independent oracle. All 50 engines participate.

The concrete suite records result, final-state, counter, and combined evidence
checksums. A sample is rejected if the application outcome differs from its
oracle, identifies the wrong scenario, reports a different operation count, or
fails validation. Repeated samples for one scenario must produce identical
semantic checksums for the fixed seed.

Only the scenario's application execution is timed. Setup, state observation,
canonicalization, checksums, and independent oracle work run after the timing
interval. This design attests the instance that was timed without counting the
validation machinery as application work.

| Profile | Operations per engine | Samples per engine | Seed | Engines |
| --- | ---: | ---: | ---: | ---: |
| `smoke` | 8 | 1 | 42 | 50 |
| `standard` | 64 | 5 | 42 | 50 |

There is no concrete `large` application profile. Operation counts have
scenario-specific meaning, and elapsed times are not comparable across unlike
engines. The suite defines no performance threshold, service-level objective,
or production capacity claim. It also does not replay every application engine
through every stable geometry backend; backend cross-checking belongs to the
generic suite.

## CI and durable artifacts

Pull requests and `main` builds run the generic smoke profile after building the
native CPU extension. Those JSON, Markdown, and checksum artifacts are retained
for 30 days.

The weekly benchmark workflow runs the generic standard sections and the
concrete application standard suite. Each generated bundle is checksum-checked
before its 90-day GitHub Actions artifact is published. Generic sections are
sampled, qualification catalog, qualification lease, qualification scheduling,
legacy applications, and payload. The concrete job writes
`applications-standard.json` and its companion files. Weekly artifacts are
retained for 90 days.

A generic section can be run directly:

```bash
uv run python -m tests.performance.benchmark_suite \
  --profile large --section qualification-catalog \
  --require-all-stable
```

A concrete subset can be selected by repeating `--scenario`:

```bash
uv run python -m tests.performance.application_benchmark_suite \
  --profile smoke \
  --scenario distributed-document-search \
  --scenario heap-free-space \
  --output build/benchmarks/applications-subset.json
```

Artifacts include the commit, runtime and platform metadata, exact profile,
scenario or workload dimensions, validation evidence, and methodology. Jobs
fail on semantic divergence, missing required backends, invalid application
samples, incomplete writes, or checksum failures. Generic and application jobs
do not fail merely because a wall-clock value changed. Attribution policy is
only enforced when a caller explicitly invokes its gate verifier.
