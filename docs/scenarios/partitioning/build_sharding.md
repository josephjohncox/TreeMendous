# Distributed build and test sharding

`BuildShardingEngine` validates a named weighted dependency DAG. It rejects duplicate names/dependencies, unknown dependencies, self-edges, nonpositive weights, and cycles. A lexical Kahn traversal creates deterministic topological order. The requested shard count is capped by task count, and a greedy remaining-weight target divides that order into nonempty contiguous shards while leaving enough tasks for later shards.

Claims cover contiguous shard IDs. `execute_claim()` checks external dependencies before recording tasks and abandons a claim on ordering failure. `run()` executes in shard order. `shards`, `snapshot()`, and `checkpoint()` expose weights, topological order, completion, and local claim/event evidence.

The engine records build completion but does not invoke compilers or tests. Its DAG and ledger are process-local. Distributed runners need an artifact store, environment/toolchain identity, durable task results, dependency notification, worker failure handling, and fencing-token validation. Contiguous shards balance estimated weights; they do not guarantee optimal makespan.

The example builds then tests in two shards. The smoke validates 100 chained tasks against an independent dependency-order oracle and checks shard contiguity.
