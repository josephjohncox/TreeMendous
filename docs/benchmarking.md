# Benchmarking and scale qualification

Tree-Mendous has reproducible benchmark profiles rather than ad hoc timing
scripts. Every profile runs the public `RangeSet` API, rejects semantic drift,
and writes durable JSON, Markdown, and SHA-256 artifacts.

## Commands

```bash
just benchmark-smoke
just benchmark-standard
just benchmark-large
```

`just benchmark` is an alias for the standard profile. Each command builds the
stable native CPU extension and requires all six stable CPU backends. Results
are written under `build/benchmarks/`:

```text
smoke.json
smoke.md
smoke.json.sha256
```

Pass a different output path as the recipe argument when retaining several
local runs:

```bash
just benchmark-smoke build/benchmarks/smoke-$(git rev-parse --short HEAD).json
```

## What is exercised

The sampled traces model four core workload shapes rather than random method
calls:

- **Fragmented allocator churn:** repeated release, reserve, lookup, atomic
  allocation, impossible fits, and invalid requests over alternating free and
  occupied spans.
- **Immutable capacity catalog:** large read-mostly sets with `first_fit`,
  `overlaps`, `snapshot`, and `stats` checkpoints.
- **Bounded compute scheduling:** short, medium, and long jobs with release
  coordinates, exclusive deadlines, occupancy pressure, cancellation, success
  by class, and Jain fairness.
- **Sharded lease pools:** port, ID, address, or seat leases across isolated
  pools, including block allocations and duplicate or stale cancellations.

Separate payload traces exercise every explicit policy:

- uniform tenant labels with payload predicates;
- commutative/idempotent access or ownership overlays;
- coordinate/event-key ordered booking overlays.

The payload layer is replayed through every stable geometry backend. The
property-based payload law suite remains the independent check for algebraic
laws; the benchmark rejects any cross-backend state or query divergence under
load.

A separate application matrix qualifies 50 concrete tasks across distributed
partition claiming, scheduling/reservation, overlap catalogs, allocation churn,
and numeric resource leasing. This includes distributed document search,
distributed regex scanning, distributed cluster scheduling, and distributed
genetic search. See the [use-case matrix](use-cases.md) for exact range semantics
and explicit non-goals.

## Profiles

| Profile | Purpose | Sampled scale | Load qualification |
| --- | --- | --- | --- |
| `smoke` | Required PR/build check | 32–128 initial ranges and 200–1,100 operations per workload, 20 independent samples | 500-range catalog, 250-shard lease pool, all payload policies |
| `standard` | Weekly engineering run | Up to 128 initial ranges and 1,100 operations, 20 independent samples | 10,000-range catalog, 2,000-shard lease pool, 25,000 scheduled jobs |
| `large` | Manual production-scale qualification | Up to 128 initial ranges and 1,100 operations, 20 independent samples | 25,000-range catalog, 5,000-shard lease pool, 50,000 scheduled jobs |

Every profile also executes all 50 application scenarios against all stable
backends: 40 operations per scenario in smoke, 100 in standard, and 200 in
large. The large profile separates interval cardinality from operation count
where a linear-scan backend would otherwise create a meaningless cross product.
It qualifies high-cardinality state and high-volume operation loads
independently, using workloads that resemble actual service behavior.

## Correctness before timing

For geometry workloads, the harness owns an independent sorted-list oracle with
its own span type, validation, binary-search index, mutation implementation, and
accounting. It does not import production interval algorithms. Before a timing
sample is accepted, an equivalent replay must match:

- every fit, allocation, overlap, snapshot, and statistics observation;
- every successful mutation, no-op, expected error, touched interval count, and
  changed length;
- the complete normalized final state and total availability;
- scheduling success by class and fairness;
- state and query SHA-256 checksums.

The timed replay contains only the declared public operations. Accounting,
normalization, serialization, checksums, and divergence rejection occur in a
separate replay and are reported as validation overhead.

Sampled measurements use at least 20 independent runs, warmups inside each
sampling worker, deterministically randomized backend order, medians, median
absolute deviation, and a 95% run-level bootstrap interval for the median.
Per-operation invocation distributions are descriptive and do not pretend that
calls from one trace are independent samples.

Large-scale qualification deliberately uses one timed replay after complete
validation. Its elapsed time and operations/second are load observations, not a
basis for ranking backends or enforcing a performance threshold.

## Durable CI runs

Every pull request and `main` build runs the smoke profile after building the
native CPU backend. GitHub Actions uploads the JSON, Markdown, and checksum
files for 30 days.

`.github/workflows/benchmarks.yml` runs the standard profile weekly and accepts
manual `standard` or `large` dispatches. Long profiles are split into sampled,
catalog, lease, scheduling, applications, and payload jobs so each section produces an
independently useful artifact and one slow workload cannot erase completed
results. The same sections are available locally:

```bash
uv run python -m tests.performance.benchmark_suite \
  --profile large --section qualification-catalog \
  --require-all-stable
```

Artifacts are retained for 90 days and include the commit,
Python/runtime/platform metadata, GitHub run provenance, exact profile and
section, backend list, dimensions, validation evidence, and methodology.

Benchmark jobs fail on semantic divergence, unavailable stable backends,
incomplete artifact writes, or checksum failures. They do **not** fail because
a wall-clock number changed. Comparing performance across different hosted
runners, CPU governors, compiler versions, or concurrent load would produce a
false regression signal.
