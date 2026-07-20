# GPU memory arena

`GPUMemoryArena` manages one contiguous device-address domain. The base address
must satisfy the device alignment, and per-buffer alignment must be a power of
two no smaller than that device minimum. Each `GPUBuffer` is owned by the stream
that allocated it.

## Deferred frees

GPU work is asynchronous, so `defer_free` does not normally return memory
immediately. It records a completion epoch on the owning stream. A wrong stream
is rejected. `advance_completion(stream, epoch)` is monotonic and reclaims all
eligible buffers in epoch/allocation order. A deferred free whose epoch is
already complete can reclaim immediately. This avoids exposing memory to a new
kernel while an older kernel may still use it.

Snapshots separate usable live buffers from deferred buffers, report completed
stream epochs, and expose actual allocator free geometry. Deferred buffers
remain counted as allocated until reclamation. Advance takes an allocator
checkpoint so an unexpected release failure restores the whole batch.
Checkpoints preserve buffer alignment, stream identity, pending epochs, and
completed epochs; restore validates them before commit.

```python
from treemendous.applications.allocation.gpu_arena import GPUMemoryArena

arena = GPUMemoryArena(1 << 20, device_alignment=256)
buffer = arena.allocate(64000, stream="render", alignment=512)
arena.defer_free(buffer, stream="render", completion_epoch=7)
assert arena.advance_completion("render", 6) == ()
assert arena.advance_completion("render", 7) == (buffer,)
```

The example demonstrates stream-ordered reclamation. The independent oracle
filters pending records by stream and epoch. The smoke allocates, defers, and
advances real arena state for every measured operation.
