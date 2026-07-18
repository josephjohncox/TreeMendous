# Canonical API

The stable entry point is `treemendous.create_range_set`. Direct classes under
`treemendous.basic` are compatibility/advanced APIs and can expose historical
result shapes. `treemendous.basic.segment` is explicitly experimental.

## Values and validation

- `Span(start, end)` validates an integer half-open range `[start, end)`.
- `ManagedDomain` normalizes adjacent spans and rejects overlapping spans.
- `IntervalResult` is an immutable query result.
- `MutationResult` reports changed geometry, changed length, and whether the
  requested span was fully covered before mutation.
- `RangeSnapshot` contains immutable intervals, total free measure, and domain.
- `AvailabilityStats` derives free/occupied measures only from an explicit
  domain. Calling `stats()` without one raises `ManagedDomainRequiredError`.

Empty/reversed spans and lengths at or below zero raise `ValueError` before any
mutation. Non-integer coordinates raise `TypeError`. Canonical mutations must
fit wholly inside one managed-domain component.

## `RangeSet`

- `add(span, payload=...)` adds available geometry.
- `discard(span, require_covered=False)` removes available geometry. With
  `require_covered=True`, incomplete coverage returns unchanged evidence.
- `first_fit(length, not_before=..., not_after=...)` chooses the earliest fit.
  `not_after` is an exclusive allocation deadline, so a result may end exactly
  at that coordinate.
- `allocate(...)` performs first-fit and coverage-checked removal under one
  re-entrant lock. It returns `None` with no mutation when no fit exists.
- `overlaps(span)`, `intervals()`, `stats()`, and `snapshot()` expose canonical
  immutable observations.

Largest-fit ties use longest then earliest. Best-fit compatibility queries use
least waste then earliest. The canonical facade currently exposes first-fit;
capability-specific legacy methods remain available through adapters.

## Payload policies

Payload support is advertised only by Python backends. When no policy is
supplied, payload-capable backends use `UniformPayloadPolicy` safely by default.
An explicit policy is needed only to request non-uniform join or ordered
semantics.

- `UniformPayloadPolicy`: equality-only adjacency and overlap. `None` is data.
- `JoinPayloadPolicy`: pointwise commutative, associative, idempotent join with
  an explicit bottom value. Those laws are caller obligations.
- `OrderedPayloadPolicy`: associative fold in stable coordinate/event-key order.

All policies may define `restrict(data, source, target)` behavior for splits.
Geometry stays canonical even when payload endpoints require segmentation.
Mutable committed values are copied around user callbacks. Native backends are
geometry-only and reject payload policies or predicates.

## Concurrency

Each facade call is lock-scoped within one `RangeSet` instance. `allocate` is
atomic only with respect to other calls through that same facade. Raw backend
objects returned by the deprecated `get_raw_implementation()` escape hatch are
not covered by this guarantee.

## Legacy migration

`create_interval_tree`, `UnifiedIntervalManager`, and these methods remain as
compatibility wrappers: `release_interval`, `reserve_interval`, `find_interval`,
`get_intervals`, and total queries. Legacy mutation wrappers return `None`;
canonical `add` and `discard` return `MutationResult`. Historical payload hooks
are accepted by direct Python engines, while the canonical facade uses its
implicit uniform policy unless an explicit join or ordered policy is supplied.
