"""Publish ordered partition-availability geometry as one atomic batch.

This pattern is outside the 50-engine registry. It has no partition identity,
replica-health or quorum model, generation/fencing semantics, durable catalog,
or distributed reconciliation protocol; those domain and persistence services
must remain external.
"""

from treemendous import MutationResult, Span
from treemendous.exact_batch import (
    BatchLimits,
    BatchMutation,
    ExactBatchRangeSet,
    MutationOpcode,
)


def main() -> None:
    limits = BatchLimits(
        max_operations=3,
        max_live_intervals=4,
        max_changed_spans=6,
        max_result_bytes=2_048,
        max_work_units=24,
    )
    available = ExactBatchRangeSet(((0, 4), (8, 16)), limits=limits)
    results = available.mutate(
        (
            BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 2, 4),
            BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 10, 14),
            BatchMutation(MutationOpcode.ADD, 12, 16),
        )
    )

    assert results == (
        MutationResult((Span(2, 4),), 2, True),
        MutationResult((Span(10, 14),), 4, True),
        MutationResult((Span(12, 14),), 2, False),
    )
    assert tuple(
        (interval.start, interval.end) for interval in available.snapshot().intervals
    ) == ((0, 2), (8, 10), (12, 16))
    assert available.limits is limits
    print("available=0:2,8:10,12:16 changed=2,4,2")


if __name__ == "__main__":
    main()
