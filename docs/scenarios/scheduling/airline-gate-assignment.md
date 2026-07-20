# Airline gate assignment

`AirlineGateScheduler` models exclusive gates, aircraft compatibility, and
turnaround occupancy. A flight's arrival/departure interval is user-visible;
`turnaround_before` and `turnaround_after` extend the occupied half-open span.
Consequently another flight may begin exactly at the prior occupied end but not
inside its handling buffer.

The deterministic policy chooses an earliest compatible gate/name within the
exact supplied service interval. Conflict diagnostics identify the gate,
overloaded segment, and blocking reservations. An invalid or conflicting
request is failure atomic. Owner-scoped request IDs replay identical input,
cancellation releases the entire buffered interval, and snapshots retain both
active and cancelled immutable records.

This is not airport operations research software. It does not model taxiways,
towing, passenger connections, gate changes, stochastic delay, airline
preferences, or solve the NP-hard global assignment problem. It provides local,
reproducible feasibility only. State has thread-safe in-process atomicity but no
cross-process transaction, durable log, or distributed lease.

Run `python examples/applications/scheduling/airline_gates.py`.
