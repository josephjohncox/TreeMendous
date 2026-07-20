# API

Tree-Mendous exposes one interval API: `RangeSet`. All intervals are half-open
`[start, end)` spans over a bounded integer domain.

## Construction

```python
from treemendous import create_range_set

ranges = create_range_set(
    domain=(0, 1_000),
    initially_available=True,
)
```

`domain` accepts a `ManagedDomain` or a two-integer tuple. `backend` is a
backend ID or `None`. Omitting it selects automatically from stable backends
whose semantic probe passed.

Use a registry when discovery must be controlled or reused:

```python
from treemendous import BackendRegistry, create_range_set

registry = BackendRegistry.discover()
ranges = registry.create((0, 100))
```

A registry is immutable. Tests can construct one from explicit specs and probe
states without importing optional native modules.

## Geometry

```python
from treemendous import Span

ranges.discard(Span(20, 30))
ranges.add(Span(40, 50))

assert ranges.first_fit(5, not_before=0).span == Span(0, 5)
assert ranges.snapshot().total_free == 990
assert tuple(result.span for result in ranges.intervals()) == (
    Span(0, 20),
    Span(30, 1_000),
)
```

The geometry operations are:

- `discard(span)`: remove a span from availability.
- `add(span, payload=...)`: add a span to availability.
- `first_fit(length, not_before=..., not_after=...)`: find the earliest fit.
- `allocate(length, not_before=..., not_after=...)`: find and discard atomically.
- `intervals()`: canonical coalesced availability as `IntervalResult` values.
- `snapshot()`: immutable intervals and total free length.
- `stats()`: aggregate availability statistics.

All mutations are failure-atomic. Invalid spans, out-of-domain spans, invalid
lengths, and payload-policy failures leave state unchanged.

## Payload policies

Payload semantics are implemented once in `RangeSet`; raw backends manage only
geometry. Pass a policy explicitly when values are required.

### Uniform payloads

`UniformPayloadPolicy` merges adjacent or overlapping spans only when their
values are equal. `None` is an ordinary value.

```python
from treemendous import Span, UniformPayloadPolicy, create_range_set

ranges = create_range_set(
    (0, 20),
    initially_available=False,
    payload_policy=UniformPayloadPolicy(),
)
ranges.add(Span(0, 5), payload="free")
ranges.add(Span(5, 10), payload="free")

assert tuple((result.span, result.data) for result in ranges.intervals()) == (
    (Span(0, 10), "free"),
)
```

### Join payloads

`JoinPayloadPolicy` combines overlap values with an associative, commutative,
and idempotent join operation.

```python
from treemendous import JoinPayloadPolicy, Span, create_range_set

policy = JoinPayloadPolicy(join=frozenset.union, bottom=frozenset())
ranges = create_range_set(
    (0, 20),
    initially_available=False,
    payload_policy=policy,
)
ranges.add(Span(0, 10), payload=frozenset({"cpu"}))
ranges.add(Span(5, 15), payload=frozenset({"gpu"}))
```

The resulting endpoint segments preserve the exact overlap:

```text
[0, 5)   {cpu}
[5, 10)  {cpu, gpu}
[10, 15) {gpu}
```

### Ordered payloads

`OrderedPayloadPolicy` uses an associative but order-sensitive operation.
Active events are folded in deterministic `(start, end, event_key)` order, not
in insertion order. Supply `event_key_fn` when application payloads need an
explicit stable ordering key; insertion permutations then produce the same
pointwise result.

```python
from treemendous import OrderedPayloadPolicy

policy = OrderedPayloadPolicy(
    combine_fn=lambda old, new: old + new,
    identity=tuple(),
)
```

Payload algebra and payload ownership are separate concerns. `RangeSet` uses
`copy.deepcopy` at storage, callback, and observation boundaries by default.
Pass `payload_cloner=` to `RangeSet`, `BackendRegistry.create`, or
`create_range_set` when an application needs a different cloning strategy. A
custom cloner for mutable payloads must return a semantically detached value;
an alias-preserving cloner deliberately opts out of isolation. Policy `bottom`
and `identity` values are cloned and owned when the `RangeSet` is constructed,
so later caller mutation cannot alter default additions. Stable backend
adapters must make each individual geometry mutation failure-atomic; callbacks
and payload staging complete before that mutation is invoked.

## Backend registry

```python
from treemendous import BackendRegistry

registry = BackendRegistry.discover()
print(registry.specs)
print(registry.states)
print(registry.available_specs())
```

`BackendSpec` describes identity, status, runtime, device, integer width, and
native capabilities. Each registry state records the semantic probe result. Backend selection is a
pure decision over those immutable values.

See [Backend support](backends.md) for the stable/experimental policy.
