# Distributed search-index merge

`IndexMergeEngine` copies search-index segments and validates every term and posting. Posting IDs must be nonnegative integers and each source list must be nondecreasing; duplicates are permitted because merge semantics remove them. The union of terms is sorted lexically and forms the claim domain. For each claimed term band, `heapq.merge` combines source streams, `dict.fromkeys` removes adjacent/global duplicates, and a strict-order check guards the published result.

`merge_claim()` produces complete term outputs, `run(band_size=...)` drains the job, and snapshots/checkpoints expose source term order, merged postings, and private coordination evidence. Empty segments/term universes, invalid keys, malformed/unsorted postings, and invalid bands fail explicitly.

Sources and outputs live in one process. Production index merging must establish immutable segment generations, persist term-band output, validate fencing tokens, atomically publish a new index generation, and garbage-collect only after readers release old generations. The engine does not claim those storage semantics.

The example merges duplicate postings across two segments. The smoke merges 100 terms and compares every output with an independent set-and-sort oracle.
