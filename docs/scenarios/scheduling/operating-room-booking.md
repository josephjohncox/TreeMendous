# Operating-room booking

`OperatingRoomScheduler` reserves a compatible room, every named staff member,
and every named equipment item for the same half-open procedure interval. These
requirements are submitted to one `ReservationLedger` call. If any item is busy,
none of the remaining resources are exposed as reserved; the prior snapshot is
unchanged. Diagnostics report which resource and reservations blocked the
candidate.

Rooms declare capability labels. Staff and equipment are exclusive named
resources. The bounded policy searches start time and room name deterministically.
Request IDs are owner scoped and reject changed replay input. Cancellation
releases the whole resource set, and snapshots are immutable ordered values.

The module is a technical scheduling primitive, not a clinical decision system.
It does not validate credentials, shifts, infection-control rules, emergencies,
procedure precedence, or patient safety, and it does not optimize an operating
suite. Those constraints must be resolved before calling the engine or encoded
by a surrounding policy layer. Atomicity and locking cover one Python process;
there is no durable database transaction or distributed lock.

Run `python examples/one_dimensional/applications/scheduling/operating_rooms.py`.
