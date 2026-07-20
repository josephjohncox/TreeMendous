from __future__ import annotations

import pytest

from treemendous.domain import Span
from treemendous.policies import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    UniformPayloadPolicy,
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
    policy = JoinPayloadPolicy[frozenset[str]](
        lambda left, right: left | right,
        frozenset(),
        restrict_fn=lambda data, source, target: calls.append((source, target)) or data,
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
