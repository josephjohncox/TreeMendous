# Maintenance window planning

`MaintenanceScheduler` validates three policy layers before reservation. Every
dependency task must already exist and remain active; the candidate cannot start
before the latest dependency end. The complete interval must fit one declared
service window and must not overlap any blackout. Finally, cumulative service
concurrency must remain available.

The bounded engine scans integer starts deterministically. A missing dependency,
cancelled dependency, blackout, window boundary, or capacity conflict is rejected
before mutation. Capacity errors retain conflict identities. Task-scoped request
IDs replay identical schedules, cancellation is idempotent, and snapshots are
immutable ordered ledger state. Requiring dependencies to exist also prevents a
new task from introducing a forward-reference cycle.

This is not a project-planning optimizer. It does not infer a dependency graph,
reschedule descendants after cancellation, assess deployment health, coordinate
services across processes, or optimize downtime. Windows and blackouts are
trusted integer inputs. The state machine provides only one-process locking and
failure atomicity; durable orchestration, approvals, fencing, and rollback of
external maintenance actions are outside its boundary.

Run `python examples/applications/scheduling/maintenance.py`.
