# Distributed document search

`DocumentSearchEngine` is a finite search job, not a renamed range workload. Construction copies documents in numeric ID order, tokenizes ASCII alphanumerics case-insensitively, and builds deterministic term-to-document posting lists. A query is normalized once and requires every distinct query token. Workers claim document **ordinals** (not assumptions about dense external IDs), intersect posting lists, filter results to their band, and merge hits by document ID.

Use `claim()` plus `search_claim()` when driving workers explicitly, or `run(shard_size=...)` for local execution. `snapshot()` exposes the normalized query, merged hits, and claimed-document count; `checkpoint()` pairs that state with claim/event kernel state. Invalid IDs, non-string text, empty inputs, and tokenless queries fail before claims are created.

The implementation is thread-safe only at the private claim kernel boundary. The index and result merge are process-local memory. A real distributed service must store the input/index durably, transmit claims, and accept results only when the claim fencing token is current. Its result store must also make retry commits idempotent. The bundled engine deliberately makes none of those durability, consensus, worker-liveness, or network claims.

See `examples/applications/partitioning/document_search.py`; the correctness smoke compares actual indexed execution with an independent exhaustive document oracle.
