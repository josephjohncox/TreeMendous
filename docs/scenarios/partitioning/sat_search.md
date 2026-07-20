# Distributed SAT search

`SatSearchEngine` is an exact finite CNF evaluator. Variables are numbered from one; positive literals require true and negative literals require false. A configurable number of low assignment bits forms the prefix ordinal domain. Each claimed prefix band enumerates every suffix, evaluates every clause, and records satisfying full assignments by integer ordinal. Thus partitions are disjoint and collectively cover all `2**variables` assignments.

`evaluate_claim()` executes one band and `run(shard_size=...)` exhausts prefixes. Results contain both ordinal and a variable-order Boolean tuple. `snapshot()` captures configuration and ordered solutions; `checkpoint()` includes claim/event state. Empty CNFs/clauses, literal zero, out-of-range literals, invalid variable counts, and invalid prefix widths are rejected.

This is intentionally a clear exhaustive engine rather than a CDCL solver. Large variable counts remain exponential. The claim ledger and solution map are local; distributed use requires immutable CNF identity, durable prefix ownership, fenced/idempotent solution commits, and cancellation after an accepted solution if only satisfiability is required.

The example searches a two-variable CNF. The smoke evaluates a six-variable job and compares solution ordinals with an independent truth-table oracle.
