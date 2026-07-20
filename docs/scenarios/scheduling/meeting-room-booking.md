# Meeting-room booking

`MeetingRoomScheduler` matches attendee count and required room features, then
reserves exactly one room. Inputs are integer local slots plus an explicit UTC
offset. The engine normalizes with `utc = local - offset` before conflict checks,
so callers in different fixed-offset zones observe the same timeline. Room UTC
offset metadata is informational.

A booking exposes the normalized `Span`, attendee count, chosen room, and
reservation. Half-open boundaries permit a meeting ending at a slot where the
next begins. Request IDs are owner scoped and idempotent. Conflicts do not mutate
state, cancellation is owner checked and idempotent, and snapshots are immutable
and deterministic.

This intentionally does not implement IANA timezone rules, daylight-saving
transitions, recurrence expansion, attendee calendars, invitations, or an
optimal room-assignment solver. Callers must convert civil time to integer slots
with an authoritative timezone library before booking. The engine is a
single-process reservation component with no database durability or distributed
locking.

Run `python examples/applications/scheduling/meeting_rooms.py`.
