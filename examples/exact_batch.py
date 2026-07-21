"""Run one exact, ordered, whole-batch-atomic geometry transaction."""

from treemendous import MutationResult, Span
from treemendous.exact_batch import (
    BatchLimits,
    BatchMutation,
    ExactBatchRangeSet,
    MutationOpcode,
)

limits = BatchLimits(
    max_operations=4,
    max_live_intervals=4,
    max_changed_spans=4,
    max_result_bytes=1_024,
    max_work_units=16,
)
ranges = ExactBatchRangeSet((0, 32), initially_available=False, limits=limits)
before = ranges.snapshot()
results = ranges.mutate(
    [
        BatchMutation(MutationOpcode.ADD, 8, 20),
        BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 10, 14),
        BatchMutation(MutationOpcode.ADD, 10, 14),
        BatchMutation(MutationOpcode.DISCARD_REQUIRE_COVERED, 8, 20),
    ]
)

assert results == (
    MutationResult((Span(8, 20),), 12, False),
    MutationResult((Span(10, 14),), 4, True),
    MutationResult((Span(10, 14),), 4, False),
    MutationResult((Span(8, 20),), 12, True),
)
assert ranges.snapshot() == before
assert ranges.limits is limits
print("changed=12,4,4,12 restored=True max_operations=4")
