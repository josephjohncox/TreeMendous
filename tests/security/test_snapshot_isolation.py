"""Mutable payload ownership regressions for RangeSet observations."""

from __future__ import annotations

from typing import Any

import pytest

from treemendous.backends.adapters import BackendAdapter
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import IntervalResult, Span
from treemendous.policies import JoinPayloadPolicy
from treemendous.rangeset import RangeSet


def _ranges(*, domain: Any = None) -> RangeSet:
    return RangeSet(
        BackendAdapter(IntervalManager()),
        domain=domain,
        initially_available=False,
        payload_policy=JoinPayloadPolicy(lambda left, right: left + right, []),
    )


@pytest.mark.parametrize(
    "observation", ["intervals", "snapshot", "overlaps", "first_fit"]
)
def test_mutable_payload_observations_are_detached(observation: str) -> None:
    ranges = _ranges()
    ranges.add(Span(0, 10), ["A"])

    if observation == "intervals":
        observed = ranges.intervals()[0]
    elif observation == "snapshot":
        observed = ranges.snapshot().intervals[0]
    elif observation == "overlaps":
        observed = ranges.overlaps(Span(1, 2))[0]
    else:
        fit = ranges.first_fit(2, not_before=0)
        assert fit is not None
        observed = fit

    assert isinstance(observed, IntervalResult)
    observed.data.append("external")
    assert ranges.intervals()[0].data == ["A"]


def test_allocate_result_is_detached_from_retained_payload_state() -> None:
    ranges = _ranges()
    ranges.add(Span(0, 10), ["A"])
    allocated = ranges.allocate(5, not_before=0)
    assert allocated is not None

    allocated.data.append("external")
    ranges.add(Span(0, 5), ["B"])

    assert all("external" not in item.data for item in ranges.intervals())


def test_callback_retained_values_cannot_mutate_committed_payload() -> None:
    retained: list[list[str]] = []

    def restrict(data: list[str], source: Span, target: Span) -> list[str]:
        retained.append(data)
        return data

    policy: JoinPayloadPolicy[list[str]] = JoinPayloadPolicy(
        lambda left, right: left + right,
        [],
        restrict_fn=restrict,
    )
    ranges = RangeSet(
        BackendAdapter(IntervalManager()),
        initially_available=False,
        payload_policy=policy,
    )
    ranges.add(Span(0, 10), ["A"])

    retained[0].append("external")
    assert ranges.intervals()[0].data == ["A"]


def test_mutable_policy_identity_is_owned_for_future_default_adds() -> None:
    bottom: list[str] = []
    ranges = RangeSet(
        BackendAdapter(IntervalManager()),
        initially_available=False,
        payload_policy=JoinPayloadPolicy(lambda left, right: left + right, bottom),
    )
    bottom.append("external")

    ranges.add(Span(0, 1))

    assert ranges.intervals()[0].data == []


def test_mutable_policy_identity_is_copied_per_initial_span() -> None:
    bottom: list[str] = []
    ranges = RangeSet(
        BackendAdapter(IntervalManager()),
        domain=((0, 2), (4, 6)),
        payload_policy=JoinPayloadPolicy(lambda left, right: left + right, bottom),
    )

    bottom.append("external")
    first = ranges.intervals()[0]
    first.data.append("observed")

    assert [item.data for item in ranges.intervals()] == [[], []]
