# Distributed trace spans

## Model

`TraceCatalog` preserves trace ID, span ID, parent span ID, service, operation, half-open timestamp range, and insertion order. Span IDs are unique within a trace, while identical timestamps across traces or services remain independent records.

`overlapping` locates concurrent activity for one trace, optionally filtered by service. `concurrency` derives identity-bearing coverage and peak overlap. `critical_path` follows explicit parent relationships and selects the parent/child chain with maximum summed span duration. Ties prefer earlier insertion order. Missing parents define roots, which supports partial trace ingestion; ancestry cycles are rejected rather than producing a misleading path.

## Mutation and lifecycle

Trace, span, service, and operation identifiers are nonempty. A span cannot parent itself. `update` may alter timing, parent, service, or operation while retaining the trace/span handle. Passing an explicit `None` clears a parent. `remove` deletes exactly one span, and `snapshot` provides immutable insertion-ordered state.

## Example

```python
root = catalog.add("t1", "root", 0, 40, parent_span_id=None,
                   service="gateway", operation="request")
catalog.add("t1", "db", 5, 30, parent_span_id="root",
            service="database", operation="select")
path = catalog.critical_path("t1")
```

Summed ancestry duration is a documented deterministic metric, not a causal correction for nested-time double counting.
