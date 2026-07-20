# Experimental optimized multidimensional box indexes

`BoxIndex2D`, `BoxIndex3D`, `BoxIndex4D`, and `BoundedBoxIndex` are
correctness-first experimental classes in `treemendous.multidimensional`. They
implement the same identity, payload-detachment, snapshot, half-open geometry,
and insertion-order contract as linear `BoxIndex`. They are not root exports,
are not `BackendRegistry` backends, and carry no universal speed claim.

## Fixed 2D, 3D, and 4D axis projections

Each fixed-dimensional index maintains, for every axis, a dynamic projection
sorted by `(box.lower[axis], handle.sequence)`. For query box `Q`, binary search
finds the prefix whose lower bounds satisfy
`entry.lower[axis] < Q.upper[axis]`. The index chooses the axis with the
smallest prefix. This is only a candidate filter: every candidate is rechecked
using the exact half-open predicate on every axis, including
`Q.lower[axis] < entry.upper[axis]`.

A sequence-ordered Cartesian projection is maintained alongside each sorted
projection. Its binary-search-tree order is handle sequence and its heap order
is the lower bound. It reports the selected lower-bound prefix directly in
handle-sequence order while pruning whole subtrees. Thus duplicates are not
collapsed, and the exact results are returned in insertion order without a
query-dependent ordering counter.

For `n` entries, fixed dimension `d`, selected prefix size `c`, and `k` exact
results, candidate selection and query are
`O(d log n + d*c + k)`, conventionally `O(d log n + c + k)` because these
classes fix `d` to 2, 3, or 4. The worst case has `c = n` and is linear in the
number of entries. Payload clone cost is additional.

The implementation deliberately prepares immutable replacement projections
before publishing them. Rebuilding and sequence-ordering those structures makes
insert, geometry update, and removal `O(d*n log n + n)` in the current Python
implementation, not `O(d log n)`. Data-only update still copies the authoritative
entry dictionary and is `O(n)`. This is an explicit correctness/rollback tradeoff,
not a speed guarantee. Projection storage is `O(d*n)` plus entries and payloads.

## Bounded sparse grid

`BoundedBoxIndex(bounds, cell_size, ...)` supports two through eight dimensions.
`bounds` is a required containing `Box`; `cell_size` is a positive integer tuple
of the same dimension. Inserted boxes, geometry replacements, and query boxes
must all be contained in `bounds`.

Grid coordinate `j` on an axis covers the intersection of `bounds` with

```text
[bounds.lower + j*cell_size, bounds.lower + (j+1)*cell_size).
```

The implementation allocates only occupied cell posting lists. An entry is
posted under its owner-scoped handle in every cell it touches. A query unions
postings for the cells it touches, deduplicates candidate handles internally,
sorts candidates by monotonic handle sequence, and then applies the exact
half-open box predicate. Equal boxes and equal payloads remain separate entries
and appear in insertion order.

If a query touches `q` cells, reads `p` postings, finds `c` distinct candidates,
and returns `k` results, its current cost is
`O(d + q + p + c log c + d*c + k)`, plus payload clones. In the worst case all
entries occupy a queried cell, so the query is linear (with ordering overhead);
a sparse grid does not guarantee selectivity. Copy-on-write mutation currently
copies the occupied-cell map, handle-cell map, and authoritative entry map, so
its honest cost includes `O(g + n + r)` for `g` occupied cells and `r` affected
postings, plus any affected posting-list copying. No universal comparison with
the linear or projection indexes follows from these bounds.

## Mandatory guardrails

The constructor and operations reject work before combinatorial allocation or
state mutation. There is no linear fallback.

| Limit | Default | Checked before |
| --- | ---: | --- |
| `max_total_cells` | 1,000,000 | constructing the index |
| `max_cells_per_entry` | 100,000 | inserting or replacing geometry |
| `max_cells_per_query` | 100,000 | enumerating query cells |
| `max_total_postings` | 1,000,000 | publishing insert/update postings |
| `max_estimated_bytes` | 256 MiB | allocating query or copy-on-write grid structures |

All limits must be positive integers (not `bool`). Total possible cells are the
product of the ceiling-divided grid shape; computing that integer does not
materialize the grid. Entry and query cell counts are computed from per-axis
ranges before `itertools.product` is consumed. The memory guard uses a
deliberately conservative, implementation-defined estimate for grid cells,
postings, handles, and transient copy-on-write state; diagnostics expose both
the retained estimate and configured limit. It is not a measurement of payload
objects or the Python allocator, so applications needing a whole-process memory
limit must enforce one externally. A rejected insert does not consume a handle
sequence, and a rejected update leaves its old entry and postings unchanged.

## Atomicity and diagnostics

Every mutation runs under the existing mutation/reentrancy lock in this order:

1. validate identity and geometry;
2. prepare a complete replacement strategy state (which may raise);
3. clone all ingress and returned payloads (which may raise);
4. prepare the replacement authoritative entry dictionary;
5. build one immutable published state containing strategy state, entries,
   version, and next sequence;
6. replace the single published-state reference.

Readers therefore observe the old complete state or the new complete state.
Cloner failure, grid-limit failure, or strategy-prepare failure cannot leave
postings and authoritative entries inconsistent.

`BoxIndexDiagnostics` is an immutable structural snapshot. Axis-projection
diagnostics report algorithm, dimension, version, entry/duplicate counts, and
projection sizes. Sparse-grid diagnostics additionally report containing bounds,
cell size, grid shape, total possible and occupied cells, posting count, and all
configured limits, retained memory estimate, and memory limit. Queries do not
mutate diagnostics. Same-thread cloner reentrancy is rejected before reacquiring
the mutation lock; unrelated writer threads wait for the current mutation and
then proceed normally.

See the executable [fixed-dimensional example](../../examples/multidimensional/core/fixed_box_indexes.py)
and [bounded example](../../examples/multidimensional/core/bounded_box_index.py).
The separate [multidimensional acceleration design](multidimensional_acceleration.md)
defines the workload, ordering, coordinate-width, batching, SIMD, Metal, CUDA,
and evidence requirements for future native query snapshots.
