"""Algebraic and insertion-permutation laws for ordered payload events."""

from __future__ import annotations

from hypothesis import given, strategies as st

from treemendous.backends import Capability
from treemendous.backends.adapters import PythonBackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import Span
from treemendous.policies import OrderedPayloadPolicy
from treemendous.rangeset import RangeSet

EVENTS = (
    (Span(0, 10), ("A",)),
    (Span(5, 15), ("B",)),
    (Span(7, 12), ("C",)),
)
CAPABILITIES = frozenset({Capability.CORE, Capability.PAYLOADS})


@given(st.permutations(EVENTS))
def test_ordered_fold_is_invariant_under_insertion_permutation(permutation) -> None:
    policy = OrderedPayloadPolicy(
        lambda left, right: left + right,
        (),
        event_key_fn=lambda value: value,
    )
    ranges = RangeSet(
        PythonBackendAdapter(IntervalManager()),
        capabilities=CAPABILITIES,
        initially_available=False,
        payload_policy=policy,
    )
    for span, payload in permutation:
        ranges.add(span, payload)
    observed = tuple((item.start, item.end, item.data) for item in ranges.intervals())
    expected = (
        (0, 5, ("A",)),
        (5, 7, ("A", "B")),
        (7, 10, ("A", "B", "C")),
        (10, 12, ("B", "C")),
        (12, 15, ("B",)),
    )
    assert observed == expected


@given(st.tuples(st.text(), st.text(), st.text()))
def test_ordered_fold_is_invariant_under_associative_regrouping(values) -> None:
    policy = OrderedPayloadPolicy(lambda left, right: left + right, "")
    left, middle, right = values
    assert policy.combine(policy.combine(left, middle), right) == policy.combine(
        left, policy.combine(middle, right)
    )
