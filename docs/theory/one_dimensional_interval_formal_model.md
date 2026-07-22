# Formal model for one-dimensional interval sets

This document specifies the extensional meaning of Tree-Mendous range sets and
the obligations imposed on every backend. The companion
[algorithm document](one_dimensional_interval_algorithms.md) explains concrete
representations and costs. This document separates four layers that are easy to
conflate:

1. **semantics**: the set of free integer coordinates;
2. **structure**: a backend's interval or tree representation;
3. **FFI evidence**: native-to-Python mutation results;
4. **publication**: immutable values returned by `RangeSet`.

The executable counterpart lives in `tests/formal/one_dimensional_model.py` and
the `tests/formal/test_one_dimensional_*.py` files.

## 1. Universe and denotation

Let `D` be a finite union of valid half-open integer spans. `D` may be
disconnected. Let `F subseteq D` be the coordinates currently free. A target
span is

```text
A = [a, b) = {x in Z | a <= x < b}, with a < b.
```

Its counting measure is `mu(A) = b - a`. A mutation is valid only when `A` is
contained in one normalized component of `D`; crossing a managed-domain gap is
invalid even if both endpoints lie within the outer bounds.

The algebra uses mathematical integers. A concrete refinement obligation is
quantified only over inputs admissible to that backend. In particular,
`cpp_boundary` requires coordinates, each span length, managed-domain measure,
free measure, and every checked aggregate update to fit signed 64-bit
arithmetic. Rejection of `[MIN_I64, MAX_I64)` therefore does not contradict the
set equations: that span is outside the native backend's admissible universe.

A domainless `RangeSet` drops the `A subseteq D` premise and accepts any
backend-admissible finite target. After every finite operation trace, its free
state is still a finite union of spans, so the same mutation and normalization
algebra applies with an implicit unbounded coordinate universe.

A raw backend may store any structure whose denotation is `F`. The stable
public `RangeSet` view is stricter: it publishes one canonical immutable tuple
of `IntervalResult` values.

## 2. Canonical normal form

Define `NF(F)` as the coordinate-ordered maximal runs of consecutive members of
`F`:

```text
NF(F) = ([s_0, e_0), ..., [s_(n-1), e_(n-1)))
```

with:

```text
s_i < e_i
s_i < s_(i+1)
e_i < s_(i+1)
union_i [s_i, e_i) = F.
```

The strict final inequality means canonical spans neither overlap nor touch.

### Existence

Sort the points of finite `F`. Start a run at the first point, extend it while
the next point is exactly one greater than the previous point, and close the
run at the first gap. The resulting spans are valid, ordered, separated, and
denote exactly `F`.

### Uniqueness

Every canonical span is a maximal connected component of `F` under integer
adjacency. The left endpoint is the least point in that component; the right
endpoint is one greater than its greatest point. Connected components are
unique, so two canonical tuples with the same denotation have identical spans
in identical order.

### Consequences

```text
NF(denotation(NF(F))) = NF(F)              (idempotence)
NF(F) = NF(G) iff F = G                    (extensional equality)
```

A backend is therefore free to retain adjacent raw nodes only if its public
refinement normalizes them before observation.

## 3. Mutation algebra and exact evidence

For a valid target `A`, release and reserve are set operations:

```text
release:
  Delta+ = A \ F
  F'     = F union A

reserve:
  Delta- = A intersect F
  F'     = F \ A
```

A committed `MutationResult` must satisfy:

```text
changed        = NF(Delta+ or Delta-)
changed_length = mu(Delta+ or Delta-)
fully_covered  iff A subseteq F before the mutation.
```

`fully_covered` describes the pre-mutation state. It is not a synonym for
"changed" or "operation succeeded."

For `reserve(require_covered=True)` when `A` is not a subset of `F`, the
candidate intersection is not observable. The transaction is rejected:

```text
F'             = F
changed        = ()
changed_length = 0
fully_covered  = false.
```

### Measure corollaries

The delta partitions `A`, so:

```text
mu(F union A) = mu(F) + mu(A \ F)
mu(F \ A)     = mu(F) - mu(A intersect F)
mu(A \ F) + mu(A intersect F) = mu(A).
```

These equations justify updating a cached total directly from exact mutation
evidence. They are checked exhaustively for every free subset and target span
of a six-coordinate universe in
`test_one_dimensional_algebra.py`.

For a nonempty canonical state with `n` components, total free measure `M`, and
largest component measure `L`:

```text
M / n <= L <= M
0 <= 1 - L/M <= 1 - 1/n.
```

In exact real arithmetic, the fragmentation expression is zero exactly when
there is one canonical free component. The public statistic is a binary
floating-point approximation, computed as `(M - L) / M` to avoid cancellation
when `L / M` rounds to one. At extreme integer magnitudes a positive ratio can
still underflow, so callers must use `free_chunks`, not floating-point equality
to zero, to decide whether the state has one component. Fragmentation is not
monotone under release or reserve, and it does not encode an application's
request-size distribution.

## 4. Structural locality and count equations

Extensional changes are confined to `A`:

```text
F symmetric_difference F' subseteq A.
```

Representation changes can extend outside `A`: an effective release can absorb
an adjacent predecessor or successor whose coordinates were already free.

### Release

Release affects the overlap-or-touch component reached from `A`. If an
effective release absorbs `k` old canonical spans, it replaces them with one:

```text
n' = n - k + 1.
```

The case `k = 0` inserts a new component and gives the maximum count increase,
`n' = n + 1`.

### Reserve

Reserve affects one consecutive block of `k` overlapping canonical spans. At
most the left edge and right edge survive, so with `r <= 2` boundary
remainders:

```text
n' = n - k + r.
```

A middle split has `k = 1`, `r = 2`, and increases the component count by one.
No mutation can increase canonical fragmentation count by more than one.

## 5. Fragmentation potential and amortized splice work

Let the potential be the canonical component count:

```text
Phi(F) = n(F).
```

Count only abstract canonical entry insertions and deletions. Exclude ordered
search, balancing rotations, result allocation, payload work, FFI conversion,
and immutable publication.

For effective release:

```text
concrete splice work = k + 1
Delta Phi           = 1 - k
amortized work      = (k + 1) + (1 - k) = 2.
```

For effective reserve:

```text
concrete splice work = k + r
Delta Phi           = r - k
amortized work      = (k + r) + (r - k) = 2r <= 4.
```

Across `m` effective mutations, total abstract splice edits are bounded by
`4m + Phi(F_0)`. This theorem does **not** amortize immutable Python tuple
publication. Repeated local mutations followed by publication can still copy
`Theta(n)` references each time and cost `Theta(mn)`.

## 6. Fit pruning as a certificate system

For a subtree `T`, define the truths:

```text
L(T) = maximum complete span length in T
E(T) = maximum span end in T.
```

Suppose maintained values are upper bounds:

```text
U_length(T) >= L(T)
U_end(T)    >= E(T).
```

For a fit request starting at `p` with length `ell`, the subtree is safely
rejected when:

```text
U_length(T) < ell or U_end(T) <= p.
```

These are rejection certificates, not success certificates. Passing both
bounds does not prove that one span is both long enough and late enough. The
maximum length may come from a span ending before `p`, while the maximum end may
come from a later span shorter than `ell`.

Underestimation of **either** field can cause a false negative. Overestimation
preserves correctness but weakens pruning. Earliest-result correctness then
requires an in-order left/node/right search that visits the leftmost subtree not
rejected by valid certificates.

Maximum-length augmentation cannot make best fit logarithmic in general. It can
reject a subtree whose maximum is too short, but it does not identify the
smallest sufficient span among all qualifying nodes.

## 7. Lower bounds

The following costs are output or information lower bounds, independent of
backend cleverness:

- Materializing a fresh tuple of `n` intervals takes `Omega(n)` time and output
  storage.
- Mutation evidence with `z` changed components takes `Omega(z)` output work.
- An overlap query returning `z` intervals takes `Omega(z)` output work.
- Without a valid subtree maximum or equivalent certificate, a failed fit query
  can require inspecting every indistinguishable candidate: `Omega(n)` in the
  worst case.

Returning an already-published immutable tuple is the exception: a valid cached
`intervals()` result can be returned in `Theta(1)`.

## 8. Publication state machine

For an authoritative payload-free backend, model the immutable geometry cache
with:

```text
V       valid published tuple
P+ / P- invalid tuple plus one patchable add/discard
D       invalid tuple; rematerialization required.
```

Transitions are:

```text
V  -- effective mutation --> P+ or P-
P  -- effective mutation --> D
V/P/D -- no-op -----------> unchanged state
P  -- intervals() --------> V by a Theta(n) tuple patch, no backend enumeration
D  -- intervals() --------> V by backend enumeration and normalization
V  -- intervals() --------> V in Theta(1), preserving tuple identity.
```

The patch rebuilds only changed `IntervalResult` objects, but tuple slicing and
unpacking still copy or scan `Theta(n)` references. Payload-bearing sets use the
payload path rather than this authoritative geometry state machine.

`snapshot()` is distinct from cached `intervals()`. The first geometry-only
snapshot for a state costs `Theta(n)` because `RangeSnapshot.__post_init__` sums
every interval; unchanged calls then return that exact cached immutable value in
`Theta(1)`. A changed state's first snapshot and every payload-bearing snapshot
remain `Theta(n)` (plus payload cloning where applicable).

The transition and enumeration claims are executed in
`test_one_dimensional_publication.py` without timing assertions.

## 9. Native prepared-transaction refinement

The stable C++ exact-delta path implements one prepared transaction per
mutation:

1. acquire a per-manager RAII mutation guard;
2. validate signed-int64 inputs and managed-domain containment;
3. perform one ordered search and affected-range traversal;
4. build exact delta components and checked measures;
5. preallocate any replacement or right-split map node as a detached node
   handle;
6. construct canonical Python `Span`, `MutationResult`, or `IntervalResult`
   values while the live map remains unchanged;
7. verify exact Python result types;
8. commit by integer assignment, compatible node-handle insertion, iterator
   erasure, and total assignment.

The map uses an explicitly `noexcept` comparator and an always-equal standard
allocator. The compatible node was already allocated, so commit performs no
allocation or Python execution. Same-manager raw mutation is rejected while a
plan exists; queries remain legal and observe the complete pre-transaction
state. A constructor exception destroys the plan, drops the detached node, and
releases the guard without changing geometry.

This is stronger than a generation check. A generation check would detect a
nested mutation only after the nested mutation had already committed.

The common one-component path still constructs a fresh `Span`. Reusing the
caller's `Span` would require a new raw-manager convention, change constructor
hook counts and identity, and make C++ behavior differ from other authoritative
adapters. Bypassing dataclass constructors would likewise weaken the specified
failure semantics.

## 10. Backend refinements

All listed backends are stable catalog members in the current repository.

### `py_boundary`

- **Raw adjacency:** coalesced.
- **Aggregates:** cached total.
- **Fit certificate:** none per subtree.
- **Structure:** SortedDict segmented ordered map.
- **Mutation/output:** bisect, indexed key access, individual deletion and
  insertion. Exact Python delta and publication costs are separate.

### `py_avl_earliest`

- **Raw adjacency:** may retain touching spans.
- **Aggregates:** height, `max_length`, `max_end`, and `min_start`.
- **Fit certificate:** maximum length and end.
- **Height:** deterministic AVL bound.
- **Mutation/output:** `RangeSet` normalizes the raw denotation; `min_start` is
  not used by the current fit search.

### `py_summary`

- **Raw adjacency:** coalesced.
- **Aggregates:** total, count, endpoints, largest span, and derived summary.
- **Fit certificate:** largest length and latest end.
- **Height:** deterministic AVL bound.
- **Mutation/output:** summary reconstruction raises mutation constants.

### `py_treap`

- **Raw adjacency:** may retain touching spans.
- **Aggregates:** subtree size and total.
- **Fit certificate:** no maximum-length certificate.
- **Height:** expected logarithmic only for fresh IID priorities.
- **Mutation/output:** split-derived priorities are not IID; dynamic histories
  retain only structural guarantees.

### `py_boundary_summary`

- **Raw adjacency:** coalesced.
- **Aggregates:** total plus lazy whole-map summary.
- **Fit certificate:** boundary scan.
- **Structure:** SortedDict segmented ordered map.
- **Mutation/output:** the first dirty summary read is `Theta(n)`; later reads
  are cached.

### `cpp_boundary`

- **Raw adjacency:** coalesced.
- **Aggregates:** cached total; largest currently scans.
- **Fit certificate:** none per subtree.
- **Height:** deterministic `std::map` bound.
- **Mutation/output:** exact delta uses one prepared traversal. The changed
  buffer stores two components inline and grows to `Theta(z)` storage only when
  necessary; Python still publishes `z` result objects.

The public refinement test instantiates every available stable backend for
every free subset and valid target in a five-coordinate universe. It checks
both canonical `RangeSet` publication and the raw adapter geometry after each
release, reserve, and covered-reserve rejection, so fallback-cache publication
cannot mask a raw mutation fault.

### Treap probability qualification

For a fixed key set with independent continuous random priorities, expected
treap height is `Theta(log n)`. `log2(n + 1)` is only an ideal balanced-tree
baseline, not the expected treap height. Tree-Mendous reservation splits derive
child priorities from the parent, so priorities after arbitrary mutation
histories are not IID. Seeded height tests are smoke tests, not proofs of the
probabilistic theorem.

## 11. Worked traces

### Bridge release

```text
F      = [0,2), [4,6), [8,10)
A      = [1,9)
Delta+ = [2,4), [6,8)
F'     = [0,10)
```

Here `k = 3`, `z = 2`, and `n: 3 -> 1`. Abstract splice work is `4`; the
potential change is `-2`; amortized splice work is `2`.

### Single-span split

```text
F      = [0,10)
A      = [3,7)
Delta- = [3,7)
F'     = [0,3), [7,10)
```

Here `k = 1`, `r = 2`, and `n: 1 -> 2`. This witnesses the maximum `+1`
component-count increase.

### Partial reservation over gaps

```text
F      = [0,3), [5,8), [10,12)
A      = [2,11)
Delta- = [2,3), [5,8), [10,11)
```

`fully_covered` is false. Ordinary reserve commits all three changed
components. `require_covered=True` returns empty evidence and leaves `F`
unchanged.

### Pruning corruption

A subtree containing `[5,9)` has true `L = 4` and `E = 9`. Query `(p=8,
ell=1)` succeeds. A stale underestimate `U_end = 8` incorrectly prunes it. An
overestimate `U_end = 10` remains sound but may explore unnecessarily.

## 12. Executable correspondence

- Normal-form existence, uniqueness, and idempotence:
  `test_normal_form_exists_is_unique_and_is_idempotent`.
- Delta equations, measure, locality, `k/r` count equations, and potential:
  `test_release_and_reserve_algebra_measure_and_locality`.
- Covered-reserve rejection:
  `test_require_covered_rejection_is_an_atomic_observable_noop`.
- Fragmentation bounds:
  `test_fragmentation_measure_bounds_hold_for_every_nonempty_state`.
- Stable public and raw-backend refinement plus canonical output:
  `test_one_dimensional_backend_refinement.py`.
- Pruning soundness, underestimation counterexamples, production AVL balance,
  and production augmentation recomputation after bulk deletion:
  `test_one_dimensional_augmentations.py`.
- `V/P/D` publication transitions and enumeration counts, without timing-based
  complexity claims: `test_one_dimensional_publication.py`.
- Native constructor failure and raw reentrancy:
  `tests/unit/test_native_core_contract.py`.

The finite universe does not prove unbounded implementation correctness by
itself. It makes the algebra executable, catches boundary mistakes, and links
each implementation obligation to a deterministic regression test. Property
and state-machine tests cover longer generated traces.
