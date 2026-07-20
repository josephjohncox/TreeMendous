# Packet sequence reassembly

## Model

`PacketReassemblyCatalog` stores every nonempty received fragment as a stable identity with flow ID, sequence range, payload bytes, and arrival order. Retransmissions and coincident duplicates remain visible instead of becoming a union. Numeric ranges from other flows are filtered out of every flow-specific query.

`assemble(flow, start, end)` performs bounded byte assembly. For each requested sequence position, the earliest received fragment wins; later duplicate or conflicting bytes remain records but do not overwrite it. The result includes contributing fragments, contiguous gaps, identity-bearing coverage segments with count greater than one, and payload bytes. Payload is `None` whenever any gap remains, avoiding ambiguous concatenation across missing sequence numbers.

## Mutation and snapshots

Fragments require nonempty immutable bytes and valid nonnegative half-open sequence ranges. `update` corrects a sequence or payload while preserving the receive handle and insertion order. `remove` deletes one arrival, not every duplicate. `snapshot` returns all retained fragments in receive order.

## Example

```python
catalog.add("flow-1", 0, b"abc")
catalog.add("flow-1", 3, b"def")
result = catalog.assemble("flow-1", 0, 6)
assert result.payload == b"abcdef"
```

The engine deliberately does not implement wrapping TCP sequence arithmetic; callers must normalize sequence numbers into a linear integer domain.
