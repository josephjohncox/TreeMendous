# Experimental `BoxIndex` denotation

`treemendous.multidimensional` is an **experimental** namespace. It is not part
of the stable root `treemendous` API, is not selected through `BackendRegistry`,
and currently makes no performance claim. The first implementation is an
identity-preserving linear reference index.

## Coordinate and box model

For dimension `d >= 2`, coordinates are points of the integer lattice
`Z^d`. Python integers are exact and unbounded; `bool` is rejected even though
it is an `int` subclass.

For lower and upper vectors `l,u in Z^d`, a valid box requires
`l_i < u_i` for every axis and denotes the finite set

```text
[[l,u)) = { x in Z^d | for every axis i, l_i <= x_i < u_i }.
```

Its exact volume is `product_i (u_i - l_i)`. A box `A` contains `B` exactly
when `A.lower_i <= B.lower_i` and `B.upper_i <= A.upper_i` on every axis.
Boxes overlap exactly when

```text
A.lower_i < B.upper_i and B.lower_i < A.upper_i
```

on every axis. Consequently, face, edge, and corner contact are disjoint.
Predicates reject mixed dimensions instead of silently returning false.

These definitions imply overlap symmetry, containment reflexivity and
transitivity, and invariance under a common integer translation or common axis
permutation. Translation or permutation of only the query is not invariant.

## Identity-preserving index state

An index state is an insertion-ordered finite sequence

```text
I = [(h_1, B_1, p_1), ..., (h_n, B_n, p_n)].
```

Each `h_i` is an owner-scoped value identity with a monotonically increasing
sequence number. It is a lookup token, not an authorization capability or
secret. Sequence numbers begin at one and are never reused. Equal boxes and
equal payloads remain distinct entries with distinct handles. An
entry update preserves its handle and insertion position; removal addresses one
handle, never all entries with equal geometry.

`overlaps(Q)` denotes the insertion-ordered subsequence of every entry whose box
overlaps `Q`. Query ordering therefore does not depend on hashing or backend
traversal. `get`, `update`, and `remove` reject unknown, foreign, or removed
handles with `KeyError`.

Successful insert, explicit update, and removal each advance `version` exactly
once. Reads and failed mutations do not. Validation and all payload copies
needed by a mutation complete before commit. A copy or validation exception
leaves membership, ordering, version, and the next sequence unchanged.
Same-index mutation from payload `__deepcopy__` is rejected rather than allowed
to deadlock or interleave a partial commit.

## Payload and snapshot ownership

`BoxIndex(..., payload_cloner=...)` applies its cloner on insertion, explicit
replacement, reads, and snapshots. The default is `copy.deepcopy`. A cloner for
mutable values is required to return a semantically detached graph; Python
cannot enforce that contract when a custom cloner or `__deepcopy__` deliberately
returns aliases. Under that precondition, mutating an input or returned payload
does not mutate live index state. Clone cost is part of API cost. Mutation
attempts made reentrantly from a cloner on the same thread are rejected;
unrelated writer threads wait for the active mutation to finish. A cloner must
not synchronously wait for another thread that mutates the same index, because
the outer mutation intentionally retains its lock while cloning.

A `BoxIndexSnapshot` captures dimensions, version, ordered entries, and a
payload graph under one lock. Later live mutations cannot alter it, and
snapshot payload mutation cannot alter the live index. The snapshot record is
frozen, but arbitrary payload objects inside it are not recursively immutable.
Snapshot `get` and `overlaps` return further detached entries.

## Algorithm and diagnostics

The current algorithm stores entries in insertion order and scans all `n`
entries for an overlap query:

- storage: `O(n)` entries plus payloads;
- handle lookup: expected `O(1)` dictionary work plus payload copy;
- insert, update, and removal: `O(n)` to prepare the shared atomic entry-map
  replacement, plus payload copy;
- overlap query: `O(n + k)` geometric work plus copies for `k` results;
- snapshot and `entries`: `O(n)` plus payload-copy cost.

`BoxIndexDiagnostics` reports `algorithm="linear"`, dimensions, version, total
entry count, distinct box count, and duplicate entry count. It deliberately has
no mutable query counters.

The linear implementation remains the executable semantic reference for the
fixed axis-projection and bounded sparse-grid classes. Their algorithms,
complexity, guardrails, and diagnostics are specified in the
[optimized index design](optimized_box_indexes.md). All optimized classes refine
these observations and pass independent finite point-set and state-machine
suites; that evidence does not imply a universal speed ranking.

## Example

```python
from treemendous.multidimensional import Box, BoxIndex

index = BoxIndex(2)  # payload_cloner=copy.deepcopy is the default
first = index.insert(Box((0, 0), (8, 8)), "first")
second = index.insert(Box((0, 0), (8, 8)), "second")
assert [entry.handle for entry in index.overlaps(Box((4, 4), (5, 5)))] == [
    first,
    second,
]
index.remove(second)
```

See the [executable example](../../examples/multidimensional/core/linear_box_index.py).

## Non-goals of this slice

`BoxIndex` is a record index, not a multidimensional availability region. It
does not provide union/coalescing, subtraction, complement, first-fit,
allocation, packing, nearest-neighbor search, serialization, managed domains,
payload overlay algebra, backend selection, native acceleration, CUDA, or
Metal. Those require separate explicit semantics and correctness gates; they
must not be inferred from the identity-preserving record API.
