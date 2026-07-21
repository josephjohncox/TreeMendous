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
application examples plus the basic and multidimensional examples from an
unrelated working directory.

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
