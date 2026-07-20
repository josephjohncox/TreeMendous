# Distributed regular-expression scan

`RegexScanEngine` scans immutable bytes with a compiled **bytes** regular expression. Workers claim non-overlapping core byte chunks. Each scan expands its core left and right by the configured `halo`; a boundary match is emitted only by the chunk containing its start offset. Absolute `(start, end, value)` identities are merged and deduplicated, so overlap never duplicates a result.

The halo is explicit because Python regular expressions do not expose a safe finite maximum width for every construct. Callers must choose a halo at least as wide as matches they intend to detect. Empty-match patterns are rejected because assigning zero-width boundary ownership is ambiguous. Invalid patterns, negative halos, empty buffers, and non-byte inputs fail during construction. `snapshot()` returns ordered matches and `checkpoint()` includes local claim/event evidence.

This engine partitions byte work; it does not distribute files or provide a shared result database. Source bytes, claims, and deduplication state live in one process. A distributed adapter must provide immutable input identity, durable claims, current-token fencing on commits, and a policy for patterns whose width exceeds the halo.

The deterministic example forces a match across a chunk boundary. The smoke benchmark scans real chunks and compares them with an independent whole-buffer `re.finditer` oracle.
