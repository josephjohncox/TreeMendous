# Video edit regions

## Model

`VideoEditCatalog` stores each cut, color grade, overlay, transition, or other effect as a stable half-open frame-range identity. The payload retains region ID, track, effect name, and sorted immutable parameters. Coincident effects and edits on separate tracks remain distinct.

`regions` returns intersecting records with optional track and effect filters. `invalidation` returns both those exact records and a canonical coverage projection clipped to the requested frame window. Coverage segments retain the set of handles responsible for each segment, so a render system can distinguish overlapping effects instead of observing only their union.

## Mutation and validation

IDs, tracks, and effects must be nonempty strings. Parameters map nonempty string names to string values and are normalized into deterministic sorted tuples. Frame intervals are validated as nonempty half-open integer spans. `update` preserves identity while changing timing, track, effect, or parameters. `remove` deletes one edit. `snapshot` retains deterministic insertion order.

## Example

```python
catalog.add("grade", 0, 240, track="v1", effect="color",
            parameters={"lut": "daylight"})
affected = catalog.invalidation(100, 150, tracks=frozenset({"v1"}))
```

Invalidation reports catalog coverage only; dependency propagation to cached render products belongs to the caller.
