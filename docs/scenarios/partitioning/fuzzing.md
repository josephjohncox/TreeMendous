# Distributed fuzzing

`FuzzingEngine` maps each case ordinal to bytes using a private PRNG seeded from `(seed, ordinal)`. Input generation therefore does not depend on shard size, worker, retry, or execution order. The injected target is called for every claimed ordinal. Target exceptions are findings: their module-qualified type and message are SHA-256 hashed into a stable signature, and only the earliest ordinal for each signature is retained. Infrastructure failure is separate: it abandons the claim so the same ordinals can be retried.

`input_for()` exposes deterministic generation, `execute_claim()` runs a band, and `run(..., fail_first_claim=True)` can exercise retry behavior without pretending target crashes are worker loss. Snapshots include executed ordinals, deduplicated crashes, and retry count; checkpoints include claim/event state. Counts, sizes, seed, target, and ordinal boundaries are validated.

The target, findings, and claim ledger run in one process. Distributed fuzzing needs sandboxing, timeouts, durable corpus/findings, stable environment identity, worker-failure detection, and fencing-token enforcement. This engine intentionally provides none of those OS or cluster services.

The example shows one deduplicated exception and an abandoned claim retry. The smoke runs 400 target calls and compares every generated input plus crash signature with an independent oracle.
