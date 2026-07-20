# Laboratory equipment booking

`LabInstrumentScheduler` books an exclusive named instrument only when its
capability labels satisfy the experiment and the complete service-plus-cleanup
occupancy fits inside one calibration window. Cleanup extends occupancy after
the visible experiment interval and therefore blocks a following experiment.
The policy chooses integer start then instrument name deterministically.

Calibration windows are explicit half-open `Span` values. Failed capability,
calibration, cleanup, or capacity checks do not modify state. Conflict details
identify blocking reservations when temporal capacity is the cause. Request IDs
provide owner-scoped idempotency; cancellation is checked and repeatable; and
snapshots include immutable reservation history.

The engine does not assess calibration quality, operate hardware, schedule
technicians, predict experiment duration, or optimize throughput. Capability
strings and calibration windows are trusted caller data. It offers thread-safe
in-memory atomicity only, with no durable audit trail, distributed coordination,
or recovery after process failure.

Run `python examples/one_dimensional/applications/scheduling/lab_instruments.py`.
