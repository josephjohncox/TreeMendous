# Ring-buffer sequence capacity

`RingBuffer` combines bounded producer/consumer capacity with
`RingSequenceTracker` modular unwrapping. Cursors live in unwrapped integer
space; each result also reports its modular start and epoch. Capacity cannot
exceed the sequence modulus, ensuring outstanding slots have distinct modular
identities.

## Full policies and wraps

`produce(count, epoch_hint=...)` reserves consecutive producer sequences. An
epoch hint, when supplied, must identify the current producer cursor. Every
modular observation is recorded with its computed epoch, so wrap is explicit
and delayed values cannot be silently reclassified. `consume(count)` advances
only through available entries.

Under `backpressure`, insufficient free slots raise `RingFullError` without
moving either cursor or sequence history. Under `overwrite`, the producer
advances and the consumer is moved by exactly the overflow, recording unread
entries lost. Producing more than capacity is legal only under overwrite; the
last `capacity` entries remain resident.

Snapshots report cursors, occupancy, free slots, overwrite count, and the
tracker's received/contiguous geometry. Checkpoints include tracker state and
cursors. Restore verifies that occupancy is bounded and the contiguous history
ends exactly at the producer cursor before committing.

```python
from treemendous.applications.allocation.ring_buffer import FullPolicy, RingBuffer

ring = RingBuffer(4, sequence_modulus=8, initial_sequence=6,
                  full_policy=FullPolicy.OVERWRITE)
ring.produce(3, epoch_hint=0)  # sequences 6, 7, 0
result = ring.produce(3, epoch_hint=1)
assert result.overwritten == 2
ring.consume(2)
```

The executable example crosses a wrap. The oracle independently updates cursor
arithmetic. The smoke repeatedly performs actual production, modular tracking,
and consumption.
