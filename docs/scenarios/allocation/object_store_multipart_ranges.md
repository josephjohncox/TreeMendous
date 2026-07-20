# Object-store multipart ranges

`MultipartUploadTracker` is not a generic free-space allocator facade. It gives
one upload a fixed byte-part grid, one-based part numbers, ETag identity,
completion attempts, missing byte ranges, and contiguous-prefix completion.
The final part may be shorter than `part_size`; all earlier parts have exactly
the configured size.

## Completion and retry semantics

`complete_part(part_number, etag, size=...)` derives the only legal byte range
from the part number. Repeating the same ETag is idempotent. A different ETag
is a conflict unless `retry=True`; an explicit retry retains geometry and
increments the attempt counter. Retrying a part that never completed is also a
conflict. These rules prevent an accidental response replay from silently
changing part identity.

Completed part spans reserve exact byte ranges in `ContiguousAllocator`, whose
free view becomes `missing_ranges`. `contiguous_completion` advances only
through consecutively completed parts from byte zero, so completing part two
first does not claim a prefix. Diagnostics report total/completed parts,
missing bytes, and retry count. Checkpoint restore validates canonical part
ranges, ETags, attempts, upload ownership, and allocator handles before commit.

```python
from treemendous.applications.allocation.multipart_upload import MultipartUploadTracker

upload = MultipartUploadTracker(23, 10, upload_id="video")
upload.complete_part(2, "etag-b")
assert upload.snapshot().contiguous_completion is None
upload.complete_part(1, "etag-a")
assert upload.snapshot().contiguous_completion.end == 20
upload.complete_part(2, "etag-b2", retry=True)
```

The executable example shows out-of-order completion and retry. The independent
oracle computes state from part-number sets only. The smoke completes actual
parts and verifies the engine's completed counter.
