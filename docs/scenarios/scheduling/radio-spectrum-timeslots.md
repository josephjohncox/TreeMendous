# Radio spectrum timeslots

`RadioSpectrumScheduler` is the one scheduling scenario that requires exact 2D
geometry. Each transmission is an axis-aligned half-open rectangle whose first
axis is integer channels and second axis is integer time. `BoxIndex` performs
identity-preserving overlap queries. Guard channels expand each reservation on
both sides, clipped to the managed channel domain, so guarded rectangles cannot
intersect. Touching channel or time boundaries remain non-overlapping.

`SpectrumConflictError` returns the requested guarded box and sorted identities
of every intersecting reservation. A conflict is detected before insertion, so
failure is atomic. Owner-scoped request IDs are idempotent, cancellation removes
active geometry while retaining immutable history, and a snapshot contains both
reservation history and detached index geometry.

This rectangle model is not an RF optimizer. It does not calculate propagation,
intermodulation, power, modulation, adjacent-channel leakage, geographic reuse,
or regulatory compliance. Expanding both reservations is a conservative integer
guard policy, not a physical interference prediction. `BoxIndex` is an O(n)
experimental in-memory index, and neither it nor this scheduler provides durable
or distributed spectrum coordination.

Run `python examples/applications/scheduling/radio_spectrum.py`.
