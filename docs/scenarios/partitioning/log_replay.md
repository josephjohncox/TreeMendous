# Distributed log replay

`LogReplayEngine` consumes uniquely offset `ReplayEvent` values. Supported operations are `set`, integer `increment`, and `delete`. Inputs are sorted by offset before positional claims are created. Applied offsets form the idempotency set. After each claim, materialized state is rebuilt in global offset order, so worker completion order cannot change set/delete semantics. A failed type transition rolls back the band's offsets, rebuilds prior state, and abandons its claim.

`apply_claim()` handles explicit offset windows and `run()` drains them locally. The checkpoint contains sorted state, applied offsets, and private claim/event state. Duplicate/negative offsets, invalid operations, missing values, wrong value types, empty event sets, and increments of strings are rejected or rolled back explicitly.

All values and idempotency evidence are memory-resident. A production replay service must persist source identity, consumer checkpoints and materialized writes transactionally, fence stale consumers, and define cross-partition ordering. The engine models those semantics but is not a durable log consumer.

The example replays out-of-order input to `count=3`. The smoke processes 500 offsets and compares state with an independent sequential replay oracle.
