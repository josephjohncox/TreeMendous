# One-dimensional interval algorithms

Tree-Mendous represents free integer coordinates as a canonical set of half-open
spans. The implementations differ in how they index those spans, which
aggregates they maintain, and where they spend work. This implementation guide
complements the [formal model](one_dimensional_interval_formal_model.md). It does
not claim that one backend wins every workload.

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

Raw adjacency behavior is backend-specific. Boundary maps and `py_summary`
coalesce touching spans. `py_avl_earliest` deliberately treats equality as
non-overlap and may retain adjacent nodes; treap representations may do the
same. `RangeSet` normalizes every raw backend state before it becomes observable
through the stable API.

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
start. The algorithm then moves forward through the affected neighborhood. For
an unmanaged iterator-based balanced tree such as `std::map`, local release or
reserve is `O(log n + k)`, earliest fit is `O(log n + s)` and worst-case `O(n)`,
maintained total measure is `O(1)`, and full enumeration is `O(n)`. A managed
mutation first locates the containing domain component, so the stable C++ path
and public managed-domain validation add `O(log d)`: the complete mutation
bound is `O(log d + log n + k)`.

Those bounds must not be copied mechanically onto `py_boundary`. Its
`sortedcontainers.SortedDict` path performs one bisect, repeated rank-based
`keys()[index]` access, `k` individual deletions, and one or two insertions.
SortedContainers documents these operations as approximately logarithmic, but
its segmented lists also incur block shifts and occasional split/merge work.
The useful rigorous description is the ordered-map primitive count; the useful
engineering description is the measured cost at the configured load factor. A
local mutation usually touches a predecessor, one or two current spans, and one
insertion, which is why the plain Python boundary map can outperform more
elaborate Python trees despite weaker worst-case implementation bounds.

The worst case still matters. A release covering the whole domain touches all
`n` spans. A failed fit query can inspect all spans when no subtree or global
maximum proves that the request is impossible.

## The stable C++ boundary map

`cpp_boundary` stores the same `start -> end` relation in
`std::map<int64_t, int64_t>`. It uses checked signed 64-bit coordinate and
measure arithmetic.

The Python exact-delta binding uses preview-then-mutate. It validates and
converts the scalar arguments, asks the manager to preview the exact changed
components without changing the map, constructs the Python `Span` and
`MutationResult` objects, and only then calls the ordinary release or reserve
mutation. Consequently Python result-construction failure occurs before the
mutation. This is not a prepared transaction: the mutation performs its own
ordered search and affected-range traversal after the preview.

The ordinary mutation path itself does not copy the complete map. Before its
commit it validates managed-domain containment, computes checked measure
changes, and allocates any required insertion node. Its commit uses assignment,
iterator erasure, and node insertion. Failed validation, checked arithmetic, or
required allocation leaves the map and cached total unchanged. Common local
mutation work is `O(log d + log n + k)` with constant mutation auxiliary state,
but the exact-delta binding adds the separate preview traversal.

Exact evidence is not constant-size: a delta with `z` components stores
`Theta(z)` coordinate pairs in a native vector and publishes a Python tuple
containing `z` `Span` objects. `get_intervals()` similarly creates a native
vector of `n` pairs before pybind11 materializes Python values, so it requires
`Theta(n)` time and output storage.

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
recompute the summaries bottom-up. Both fields are pruning upper bounds:
underestimating either `max_length` or `max_end` can cause a false negative;
overestimating either remains sound but defeats pruning. `min_start` is
maintained but is not used by the current fit search.

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

For `q` summary reads between mutations in managed use, the aggregate summary
cost is `O(n + q)`: one `O(n)` recomputation followed by `O(1)` cache hits.
Alternating one mutation with one summary read remains `O(n)` per pair.

Unmanaged raw use also tracks the union of every released region as a separate
ordered boundary map. Its first analytics read materializes an immutable
`ManagedDomain` from `d` inferred components as well as scanning `n` free spans,
so that read is `O(n + d)` and `q` reads cost `O(n + d + q)`. Materialization is
deferred until analytics need it, so constructing many disjoint spans does not
rebuild a complete domain tuple after every release.

The reported fragmentation value is the floating-point evaluation of

```text
(total_free_measure - largest_free_span) / total_free_measure
```

when free measure is nonzero. This algebraically equivalent form avoids
cancellation when the largest-span ratio rounds to one, although binary-float
underflow remains possible at extreme integer magnitudes. It measures how
concentrated free capacity is. It does not incorporate the application's
request-size distribution and is not a universal allocation-failure predictor.

## Treaps

A treap combines two orders:

1. span starts satisfy the binary-search-tree invariant;
2. random priorities satisfy a heap invariant.

For a fixed key set with independent continuous priorities, expected height is
`Theta(log n)` and the worst case remains linear. `log2(n + 1)` is only an ideal
balanced-tree baseline, not the treap's expected height. Reservation splits in
the current implementations derive child priorities from the parent, so active
priorities after arbitrary mutation histories are not IID. The theorem applies
to fresh IID insertion states; seeded dynamic-height tests are structural smoke
tests rather than proof of the probabilistic guarantee. `py_treap` also stores
subtree size and total measure, which support rank and random-sampling
operations.

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

`RangeSet` reads one normalized backend snapshot during construction. Backends
with the authoritative-geometry capability return exact mutation deltas and
answer fit, overlap, allocation, and structural-statistics queries directly.
For those payload-free sets, a successful mutation updates the cached total and
invalidates a read-through interval snapshot; it does not immediately publish a
copied Python interval tuple. If the next operation observes geometry,
`RangeSet` patches the cached tuple from that single mutation. The patch rebuilds
only changed `IntervalResult` objects, but tuple slicing and unpacking still copy
or scan `Theta(n)` references. Several effective mutations without an
intervening observation instead trigger one authoritative backend snapshot when
geometry is next requested. Repeated valid `intervals()` calls return the same
cached tuple in `Theta(1)`.

Fallback backends and payload-bearing sets retain the immutable Python geometry
cache. Bisect locates a local change in `O(log n)`, but publishing a successful
fallback mutation copies tuple references around the changed slice, which is
`Theta(n)`. The fallback preserves one common semantic implementation for
payload algebra and for backends without exact deltas. `snapshot()` is a
separate cost: `RangeSnapshot.__post_init__` currently recomputes the interval
sum, so every snapshot is `Theta(n)` even when the tuple cache was already
valid.

The stable C++ boundary backend also accepts the normalized managed domain at
construction. It validates geometry mutations and overlap queries against that
domain inside the same native call. `RangeSet` still owns and exposes the public
`ManagedDomain`; moving its hot-path containment check does not transfer API
ownership to the backend.

Payload policies add their own costs. A structural policy can require segment
splits, restrictions, combinations, event ordering, and defensive clones. No
geometry-backend throughput number predicts payload-heavy throughput.

## Fragmentation controls performance

Fragmentation turns one free span into many. It increases `n`, which affects:

- tree height and ordered-map search;
- fallback tuple publication or authoritative snapshot rematerialization;
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
4. **No repeated materialization.** Exact backend deltas avoid rebuilding
   Python interval objects after each mutation. One observed local mutation is
   patched into the cached tuple; a burst of unobserved mutations incurs one
   backend rematerialization when finally observed.
5. **Staged commit.** Validation, checked arithmetic, callback work, and result
   construction precede geometry mutation. The stable C++ boundary path also
   allocates any required map nodes before its no-throw commit.
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

- For high-rate local geometry churn, start with `py_boundary` or
  `cpp_boundary`; both provide compact local boundary updates.
- For portable Python with lazy analytics, try `py_boundary_summary`; it pairs
  boundary-map mutation with cached read bursts.
- For repeated earliest-fit searches, try `py_avl_earliest`; maximum-length and
  endpoint summaries provide pruning certificates.
- For rich analytics and best-fit exploration, try `py_summary`; it maintains
  composable per-node summaries.
- For rank or random interval sampling, try `py_treap`; it maintains subtree
  sizes.

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
