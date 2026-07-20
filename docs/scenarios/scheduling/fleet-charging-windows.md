# Fleet charging windows

`FleetChargingScheduler` models exclusive chargers with integer energy delivered
per slot and connector compatibility. Required duration is ceiling division of
requested energy by charger power. A candidate must fit both the vehicle's
arrival/departure dwell and the configured maximum session bound.

The deterministic bounded policy selects earliest completion, then earliest
start, then charger name. This favors faster feasible completion without
claiming a fleet-wide optimum. Conflicts include blocking reservation IDs.
Failed dwell, connector, bound, and capacity checks leave state unchanged.
Owner-scoped request IDs replay identical sessions, cancellation releases the
charger, and snapshots expose immutable ordered reservations.

Power and energy are caller-defined integer units. The model deliberately omits
charging curves, state-of-charge uncertainty, grid capacity, tariffs,
vehicle-to-grid flows, degradation, and route optimization. It is not an energy
management system or distributed charger protocol. Locking and atomicity cover
one process; persistence and charger fencing belong to a control plane.

Run `python examples/one_dimensional/applications/scheduling/fleet_charging.py`.
