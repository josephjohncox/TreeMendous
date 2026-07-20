# Distributed cluster scheduling

`ClusterScheduler` is a reusable, process-local job placement engine. Nodes have
immutable `CapacityVector` values (for example CPU and memory) and exact-match
labels. A request supplies the same capacity dimensions, required labels, a
positive duration, and an integer release/deadline window. The policy searches
integer starts in ascending order and node names in lexical order. This makes
placement reproducible; it does **not** claim optimal bin packing, fairness, or
a globally optimal cluster schedule.

Capacity is cumulative. Two jobs may share a node and time window when their
sum fits every dimension. A failed multi-dimensional request changes no ledger
state. `SchedulingUnavailableError` reports considered resources and, when the
failure is temporal capacity, the conflicting reservation IDs and overloaded
segments. Owner-scoped request IDs replay the original reservation, while reuse
with different inputs is rejected. `cancel` is owner checked and idempotent;
`snapshot` returns immutable, deterministically ordered state.

This engine coordinates threads in one Python process. It is not a distributed
scheduler despite the scenario name: it has no leader election, fencing,
heartbeats, durable queue, preemption, gang scheduling, autoscaling, or recovery
from process loss. Persist snapshots externally only as observations; the engine
does not make them durable transactions.

Run `python examples/one_dimensional/applications/scheduling/cluster.py` for a
deterministic placement.
