# Render-farm frame scheduling

`RenderFarmScheduler` keeps two related facts: a contiguous half-open frame
chunk and a worker-time reservation. Worker concurrency is cumulative, while
active chunks for the same render may not overlap. The caller supplies render
duration; the engine does not estimate it from scene complexity.

`assign_chunk` deterministically chooses the earliest worker slot and checks
frame overlap before committing. Request IDs replay an identical assignment.
`retry` first obtains a replacement worker-time reservation and only then
retires the failed attempt, so an unavailable retry leaves the original active.
History distinguishes active, cancelled, and retried attempts. Cancellation is
owner checked and idempotent, and `snapshot` returns immutable chunk history plus
the underlying capacity schedule.

This is an in-memory allocation engine, not a renderer or distributed work
queue. It does not upload assets, detect worker death, persist outputs, choose an
optimal chunk size, or provide exactly-once rendering. A production controller
must durably record work results and reconcile worker leases. The deterministic
policy provides reproducibility rather than optimal makespan.

Run `python examples/one_dimensional/applications/scheduling/render_farm.py`.
