from __future__ import annotations

from copy import deepcopy

import pytest

from treemendous import create_range_set
from treemendous.domain import Span
from treemendous.policies import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    UniformPayloadPolicy,
)


class ExistingPayloadPolicy:
    """Structural payload policy implementing the original public contract."""

    def can_merge(self, left: list[str], right: list[str]) -> bool:
        return left == right

    def combine(self, left: list[str], right: list[str]) -> list[str]:
        if left != right:
            raise ValueError("payloads differ")
        return left

    def restrict(self, data: list[str], source: Span, target: Span) -> list[str]:
        if not source.contains(target):
            raise ValueError("target must be contained")
        return data


def test_original_structural_policy_contract_remains_supported() -> None:
    policy = ExistingPayloadPolicy()
    payload = ["A"]
    ranges = create_range_set(
        (0, 4),
        backend="py_boundary",
        initially_available=False,
        payload_policy=policy,
    )

    ranges.add(Span(0, 2), payload)
    payload.append("caller mutation")
    observed = ranges.intervals()
    assert observed[0].data == ["A"]
    observed[0].data.append("read mutation")
    assert ranges.intervals()[0].data == ["A"]


def test_payload_cloning_is_an_explicit_rangeset_concern() -> None:
    cloned: list[list[str]] = []

    def cloner(value: list[str]) -> list[str]:
        result = deepcopy(value)
        cloned.append(result)
        return result

    ranges = create_range_set(
        (0, 4),
        backend="py_boundary",
        initially_available=False,
        payload_policy=ExistingPayloadPolicy(),
        payload_cloner=cloner,
    )
    ranges.add(Span(0, 2), ["A"])

    assert ranges.intervals()[0].data == ["A"]
    assert cloned
    with pytest.raises(TypeError, match="payload_cloner must be callable"):
        create_range_set(
            (0, 4),
            backend="py_boundary",
            payload_policy=ExistingPayloadPolicy(),
            payload_cloner=None,  # type: ignore[arg-type]
        )


def test_uniform_policy_copies_splits_and_rejects_conflicts() -> None:
    source = Span(0, 10)
    target = Span(2, 5)
    payload = ["A"]
    policy = UniformPayloadPolicy[list[str]](copy_on_split=True)

    restricted = policy.restrict(payload, source, target)

    assert restricted == payload
    assert restricted is not payload
    assert policy.can_merge(["A"], ["A"])
    with pytest.raises(ValueError, match="uniform payloads differ"):
        policy.combine(["A"], ["B"])


def test_join_policy_exposes_explicit_algebra_and_restriction() -> None:
    calls: list[tuple[Span, Span]] = []

    def record_restriction(
        data: frozenset[str], source: Span, target: Span
    ) -> frozenset[str]:
        calls.append((source, target))
        return data

    policy = JoinPayloadPolicy[frozenset[str]](
        lambda left, right: left | right,
        frozenset(),
        restrict_fn=record_restriction,
    )
    source = Span(0, 10)
    target = Span(2, 5)

    assert policy.combine(frozenset({"A"}), frozenset({"B"})) == frozenset({"A", "B"})
    assert policy.restrict(frozenset({"A"}), source, target) == frozenset({"A"})
    assert len(calls) == 1
    assert calls[0][0] == source
    assert calls[0][1] == target


def test_ordered_policy_has_stable_default_and_custom_event_keys() -> None:
    policy = OrderedPayloadPolicy[str](
        lambda left, right: left + right,
        "",
        restrict_fn=lambda data, _source, target: f"{data}:{target.start}",
    )
    key = policy.event_key("A")
    assert key[0] == "builtins"
    assert key[1] == "str"
    assert policy.combine("A", "B") == "AB"
    assert policy.restrict("A", Span(0, 10), Span(2, 5)) == "A:2"

    custom = OrderedPayloadPolicy[str](
        lambda left, right: left + right,
        "",
        event_key_fn=str.lower,
    )
    assert custom.event_key("B") == "b"


@pytest.mark.parametrize(
    "policy,payload",
    [
        (UniformPayloadPolicy(), "A"),
        (JoinPayloadPolicy(lambda left, right: left + right, ""), "A"),
        (OrderedPayloadPolicy(lambda left, right: left + right, ""), "A"),
    ],
)
def test_every_policy_rejects_restriction_outside_source(policy, payload) -> None:
    with pytest.raises(ValueError, match="must be contained"):
        policy.restrict(payload, Span(0, 10), Span(9, 12))
