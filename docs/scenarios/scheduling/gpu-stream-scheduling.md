# GPU stream scheduling

`GPUStreamScheduler` models devices with cumulative `CapacityVector` limits and
named streams. Both a device and one of its streams must advertise the requested
compatibility label. A kernel reserves device capacity and the exclusive stream
in one `ReservationLedger` transition, so a failure cannot expose only half of
the placement.

Dependency inputs are readiness timestamps, not mutable graph nodes. The kernel
cannot start before the maximum readiness time. The bounded policy then scans
integer starts, device names, and stream names deterministically. Conflicts name
the existing reservations responsible for unavailable capacity. Request IDs
provide owner-scoped idempotency; cancellation releases both resources; and an
immutable snapshot includes the complete reservation history.

The model is useful for admission control and reproducible simulations. It does
not launch kernels, inspect CUDA events, infer dependencies, model warp-level
occupancy, guarantee deadlock freedom, or optimize a DAG. Integer capacity and
time are caller-normalized units. The search is deliberately bounded and makes
no NP-hard scheduling optimality claim. State and atomicity are local to one
Python process, not distributed across driver processes or machines.

Run `python examples/one_dimensional/applications/scheduling/gpu_streams.py`.
