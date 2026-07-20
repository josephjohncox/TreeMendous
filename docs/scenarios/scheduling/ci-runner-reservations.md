# CI runner reservations

`CIRunnerScheduler` reserves integer runner-time windows. Each runner declares
labels and a positive concurrency. Jobs require a label subset, duration,
release time, and deadline. Selection is earliest feasible slot followed by
runner name, which is deterministic but not a fairness or makespan optimizer.

Concurrency is enforced cumulatively by `ReservationLedger`: up to the runner's
configured slot count can overlap, and the next job moves to a later slot.
Failures return `SchedulingUnavailableError` diagnostics without changing the
snapshot. Owner-scoped request IDs are idempotent and reject changed replay
parameters. Cancellation is owner checked, repeated cancellation is safe, and
released concurrency can be reused immediately. Snapshots are immutable and
ordered.

The engine does not execute builds, interpret CI configuration, manage secrets,
preempt jobs, provide priority queues, or coordinate multiple schedulers. Labels
are literal strings rather than an expression language. Deadlines bound the
local deterministic scan. Durability, worker health, and distributed fencing
belong in a surrounding control plane.

Run `python examples/one_dimensional/applications/scheduling/ci_runners.py`.
