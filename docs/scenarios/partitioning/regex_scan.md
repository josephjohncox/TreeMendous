# Distributed regular-expression scan

`RegexScanEngine` scans immutable bytes with a compiled **bytes** regular expression. Workers claim non-overlapping core byte chunks. Every claim evaluates against the original full buffer and emits only matches whose start belongs to its core. This preserves Python semantics for anchors, word boundaries, and lookarounds instead of turning chunk or halo edges into artificial string boundaries. Absolute `(start, end, value)` identities are merged and deduplicated.

`halo` remains explicit audit configuration for integrations that implement a bounded distributed scanner, but the in-process correctness engine does not truncate regex context. Empty-match patterns are rejected because assigning zero-width boundary ownership is ambiguous. Invalid patterns, negative halos, empty buffers, and non-byte inputs fail during construction. `snapshot()` returns ordered matches and `audit_snapshot()` includes local claim/event evidence but is not restorable because it omits source bytes and pattern identity.

This engine partitions byte work; it does not distribute files or provide a shared result database. Source bytes, claims, and deduplication state live in one process. A distributed adapter must provide immutable input identity, durable claims, current-token fencing on commits, and a policy for patterns whose width exceeds the halo.

The deterministic example forces a match across a chunk boundary. The smoke benchmark scans real chunks and compares them with an independent whole-buffer `re.finditer` oracle.
