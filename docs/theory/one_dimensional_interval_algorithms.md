# One-dimensional interval algorithms

Tree-Mendous represents free integer coordinates as a canonical set of half-open
spans. The implementations differ in how they index those spans, which
aggregates they maintain, and where they spend work. This document explains the
algorithms. It does not claim that one backend wins every workload.

## Model and notation

A valid span `[a, b)` contains the integers `x` for which `a <= x < b`, with
`a < b`. Its measure is `b - a`. Two spans overlap when

```text
max(a, c) < min(b, d).
```

Touching spans such as `[0, 4)` and `[4, 9)` do not overlap. They are adjacent,
however, and the canonical free-set representation merges them into `[0, 9)`.

The canonical `RangeSet` view is the normalized sequence

```text
F = ([s_0, e_0), ..., [s_(n-1), e_(n-1)))
```

with these invariants:

1. `s_i < e_i`;
2. starts are strictly increasing;
3. `e_i < s_(i+1)`, so canonical spans neither overlap nor touch;
4. the cached total, where present, equals `sum(e_i - s_i)`.

Boundary-map and AVL implementations maintain this form directly. Some raw
experimental or treap implementations may retain adjacent entries while
representing the same union. `RangeSet` normalizes that backend state before it
becomes observable through the stable API.

The important workload variables are:

- `n`: stored free spans, which grows with fragmentation;
- `k`: spans touched by a mutation;
- `s`: spans inspected by a fit query;
- `d`: spans in the managed domain;
- `q`: queries performed between mutations.

A complexity statement that omits `k`, `s`, or the cost of publication usually
hides the expensive part.

## Release and reserve algebra

At the canonical boundary-map seam, `release([a, b))` computes
`F union [a, b)`. It finds the predecessor of `a`, merges that predecessor when
it overlaps or touches the release, merges every following span whose start is
no greater than the growing right endpoint, and publishes one maximal span.

For example:

```text
free before: [0, 3), [6, 9)
release:     [3, 6)
free after:  [0, 9)
```

`reserve([a, b))` computes `F \\ [a, b)`. Each overlapping free span is deleted,
trimmed, or split. One span can produce two remainders:

```text
free before: [0, 10)
reserve:     [3, 7)
free after:  [0, 3), [7, 10)
```

The exact changed geometry is the part of the release that was not already
free, or the part of the reservation that was free. `RangeSet` returns that
geometry in `MutationResult`; it does not reduce a mutation to a Boolean.

## Boundary maps

The boundary backends store one ordered-map entry per maximal free span:

```text
start -> end
```

The name *boundary map* refers to indexing intervals by their left boundaries.
It is not an endpoint-delta or sweep-line event map.

An ordered predecessor search locates the only span that can cross the mutation
start. The algorithm then moves forward through the affected neighborhood. At
the data-structure level:

| Operation | Work |
| --- | --- |
| local release or reserve | `O(log n + k)` ordered navigation and affected entries |
| earliest fit after a point | `O(log n + s)`, worst-case `O(n)` |
| total free measure | `O(1)` when maintained incrementally |
| enumerate all free spans | `O(n)` |

Container details alter the constants. `py_boundary` uses
`sortedcontainers.SortedDict`, whose blocked sorted lists avoid recursive Python
tree walks and keep the representation small. A local mutation usually touches
a predecessor, one or two current spans, and one insertion. That is why the
plain Python boundary map can outperform more elaborate Python trees even
though all of them have logarithmic search structures.

The worst case still matters. A release covering the whole domain touches all
`n` spans. A failed fit query can inspect all spans when no subtree or global
maximum proves that the request is impossible.

## The stable C++ boundary map

`cpp_boundary` stores the same `start -> end` relation in
`std::map<int64_t, int64_t>`. It uses checked signed 64-bit coordinate and
measure arithmetic.

The current mutation path does not copy the complete map. It first performs all
fallible work:

1. validate the request;
2. scan the affected map range;
3. compute checked changed and aggregate measures;
4. allocate a new map node if a release or right split needs one.

It then commits with integer assignment and iterator erasure. A failed
validation, checked arithmetic operation, or required allocation leaves the map
and cached total unchanged. Common local mutations therefore perform
`O(log n + k)` native work with constant auxiliary state rather than an `O(n)`
transaction copy.

Native speed comes from compiled loops, unboxed integers, and avoiding Python
object creation inside the affected-range walk. `std::map` is still a
pointer-based tree, not contiguous storage. Large scans remain limited by
pointer chasing and cache misses, and each Python call still pays pybind11
argument validation and dispatch costs.

## AVL interval trees

An AVL tree stores spans in binary-search-tree order and keeps the height of any
node's subtrees within one. Rotations restore this invariant after insertions or
deletions, which gives deterministic `O(log n)` tree height.

`py_avl_earliest` augments each node with:

- `max_length`: the longest complete span in the subtree;
- `max_end`: the greatest end in the subtree;
- `min_start`: the least start in the subtree.

Earliest-fit search uses these summaries to reject a subtree when its longest
span is too short or every span ends at or before the requested point. It then
searches in order, so the first result is the earliest valid fit. Rotations must
recompute the summaries bottom-up; a stale `max_length` can cause a false
negative, while a stale `max_end` can defeat pruning.

AVL mutation has higher Python constants than the boundary map. It allocates
node objects, follows recursive calls, updates several fields on each ancestor,
and may rotate. The augmentation earns its cost when pruned fit queries matter;
it is wasted work in a mutation-only workload.

## Summary AVL trees

`py_summary` attaches a `TreeSummary` to every AVL node. A summary combines the
left subtree, the node span, and the right subtree. Primitive fields include:

- total free measure;
- interval count;
- earliest start and latest end;
- longest free-span length and its start.

Averages, density, and fragmentation are derived from those fields. Root
analytics are available in `O(1)` after a mutation has repaired summaries along
the changed paths. Earliest fit rejects subtrees using the maximum length and
latest end. Best fit can reject subtrees whose maximum is too short, but it may
still inspect many qualifying nodes because a maximum does not encode the
smallest sufficient length.

This backend pays heavily for Python dataclass creation and summary merging on
mutation. It is appropriate when its analytics or fit queries repay that cost.
It is not the default choice for high-rate local churn.

## Boundary summaries

`py_boundary_summary` keeps the compact boundary map and a lazily computed
whole-map summary. A geometry mutation marks the summary dirty. The first
summary read scans all `n` spans; later reads return the cached value until the
next effective mutation.

For `q` summary reads between mutations, the aggregate summary cost is
`O(n + q)`: one `O(n)` recomputation followed by `O(1)` cache hits. Alternating
one mutation with one summary read remains `O(n)` per pair.

Unmanaged raw use also tracks the union of every released region as a separate
ordered boundary map. Materializing an immutable `ManagedDomain` is deferred
until analytics need it, so constructing many disjoint spans does not rebuild a
complete domain tuple after every release.

The reported fragmentation value is

```text
1 - largest_free_span / total_free_measure
```

when free measure is nonzero. It measures how concentrated free capacity is. It
does not incorporate the application's request-size distribution and is not a
universal allocation-failure predictor.

## Treaps

A treap combines two orders:

1. span starts satisfy the binary-search-tree invariant;
2. random priorities satisfy a heap invariant.

Independent random priorities give expected logarithmic height, but the worst
case remains linear. `py_treap` also stores subtree size and total measure, which
support rank and random-sampling operations.

The current treap does not store a subtree maximum span length. Earliest fit can
therefore search both subtrees and is worst-case `O(n)`. Randomization balances
the topology; it does not make an unaugmented fit predicate logarithmic.

## Canonical `RangeSet` costs

The raw geometry backend is only one layer. `RangeSet` additionally owns:

- managed-domain validation;
- mutation locking and callback reentrancy rules;
- exact `MutationResult` geometry;
- payload cloning and payload algebra;
- immutable snapshots and canonical result values.

`RangeSet` reads one normalized backend snapshot during construction and keeps
a private payload-free geometry cache synchronized after successful mutations.
Ordinary queries do not retrieve, convert, validate, and sort the entire backend
state again.

The cache is an immutable tuple. Bisect locates a local change in `O(log n)`,
but publishing a successful mutation copies tuple references around the changed
slice, which is `O(n)`. This is a deliberate publication and failure-atomicity
tradeoff. It has a low C-level copying constant, but it means the stable API's
mutation bound is not simply the raw backend's bound.

Payload policies add their own costs. A structural policy can require segment
splits, restrictions, combinations, event ordering, and defensive clones. No
geometry-backend throughput number predicts payload-heavy throughput.

## Fragmentation controls performance

Fragmentation turns one free span into many. It increases `n`, which affects:

- tree height and ordered-map search;
- tuple publication in `RangeSet`;
- failed or weakly selective fit scans;
- enumeration and snapshots;
- summary recomputation;
- memory footprint and cache locality.

Coalescing releases reduce `n`. A workload that repeatedly reserves and releases
one isolated local span can remain fast even after millions of operations,
because `k` stays small. A workload that alternates global cuts and global
merges cannot have the same cost.

Allocation size matters as much as interval count. A cached or subtree maximum
can reject a request larger than every free span immediately. Without that
summary, the algorithm must inspect every candidate to prove failure.

## Why an implementation can be very fast

High throughput follows from removing work, not from the backend label alone.
The fast path has these properties:

1. **Canonical state.** Mutations repair only a local predecessor/successor
   neighborhood instead of normalizing the whole collection.
2. **Incremental measures.** Exact changed length updates totals without a full
   scan.
3. **Prunable queries.** Maximum-length and endpoint summaries reject impossible
   subtrees.
4. **No repeated materialization.** Stable calls do not rebuild all Python
   interval objects after each mutation.
5. **Staged commit.** Validation, checked arithmetic, callback work, and required
   allocations happen before a small non-fallible publication step.
6. **Appropriate representation.** Compact boundary maps favor local churn;
   augmented trees favor the queries their summaries can prune.
7. **Amortized boundaries.** Native batch or query work must be large enough to
   repay Python-to-native dispatch, and device work must repay transfer and
   synchronization.

These conditions explain why the stable C++ boundary kernel can execute local
raw mutations much faster than the richer stable `RangeSet`, and why the plain
Python boundary map can still sustain high rates. They also explain why a
summary-heavy Python tree can be slower even with a stronger asymptotic query
story.

## Choosing a backend by workload

| Workload | Useful first choice | Reason |
| --- | --- | --- |
| high-rate local geometry churn | `py_boundary` or `cpp_boundary` | compact local boundary updates |
| portable Python with lazy analytics | `py_boundary_summary` | boundary-map mutations and cached read bursts |
| repeated earliest-fit searches | `py_avl_earliest` | maximum-length and endpoint pruning |
| rich analytics and best-fit exploration | `py_summary` | per-node composable summaries |
| rank or random interval sampling | `py_treap` | subtree-size augmentation |

This table predicts mechanisms, not winners. Run the correctness-checked
benchmark with the application's fragmentation, mutation/query ratio, domain
shape, and payload policy.

## Benchmark evidence

A valid performance result must state:

- raw backend or canonical `RangeSet`;
- backend ID and build options;
- interval count and coordinate extent;
- operation distribution, fit success rate, and fragmentation;
- payload policy and clone behavior;
- warmup, sample count, hardware, Python, and compiler versions;
- whether validation and oracle replay are inside the timed region.

The repository harness times the real public calls, then checks the same
instance's exact results, counters, queries, and final state outside timing. For
example:

```bash
uv run python -m tests.performance.protocol_benchmark \
  --operations 1000 --intervals 64 --samples 20 --warmups 2 \
  --backends cpp_boundary py_boundary py_boundary_summary \
  --output build/benchmarks/protocol.json
```

See [Benchmarking](../benchmarking.md) for artifact and interpretation rules.
