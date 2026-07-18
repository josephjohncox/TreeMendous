from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from treemendous.basic.base import IntervalNodeBase, IntervalTreeBase
from treemendous.basic.protocols import (
    AvailabilityStats,
    BackendConfiguration,
    ImplementationType,
    IntervalResult,
    PerformanceStats,
    PerformanceTier,
    standardize_availability_stats,
    standardize_interval_result,
    standardize_intervals_list,
    standardize_performance_stats,
)
from treemendous.domain import Span
from treemendous.policies import (
    JoinPayloadPolicy,
    LegacyPayloadPolicy,
    OrderedPayloadPolicy,
    UniformPayloadPolicy,
)
from treemendous.protocols import RangeSetProtocol


class _Node(IntervalNodeBase["_Node", Any]):
    def update_stats(self) -> None:
        self.update_length()


class _Tree(IntervalTreeBase[_Node, Any]):
    def _print_node(self, node: _Node, indent: str, prefix: str) -> None:
        print(f"{indent}{prefix}{node.start}:{node.end}")

    def get_intervals(self) -> list[IntervalResult]:
        return []


def test_interval_result_standardization_accepts_all_legacy_shapes() -> None:
    canonical = IntervalResult(1, 3, data="A")
    shaped = SimpleNamespace(start=7, end=9, data="B")

    pair = tuple([1, 3])
    triple = tuple([1, 3, "A"])
    interval_list = [None, tuple([0, 2]), shaped]
    assert standardize_interval_result(None) is None
    assert standardize_interval_result(canonical) is canonical
    assert standardize_interval_result(pair) == IntervalResult(1, 3)
    assert standardize_interval_result(triple) == canonical
    assert standardize_interval_result(shaped) == IntervalResult(7, 9, data="B")
    assert standardize_intervals_list(interval_list) == [
        IntervalResult(0, 2),
        IntervalResult(7, 9, data="B"),
    ]
    assert standardize_intervals_list(None) == []
    with pytest.raises(ValueError, match="Cannot standardize result"):
        standardize_interval_result((1, 2, 3, 4))
    with pytest.raises(ValueError, match="Cannot standardize result"):
        standardize_interval_result(object())


def test_statistics_standardization_accepts_values_mappings_and_objects() -> None:
    canonical = AvailabilityStats(
        total_free=8,
        total_occupied=2,
        total_space=10,
        free_chunks=2,
        largest_chunk=5,
        bounds=(0, 10),
    )
    assert standardize_availability_stats(canonical) is canonical
    from_mapping = standardize_availability_stats(
        {"total_free": 8, "free_chunks": 2, "largest_chunk": 5}
    )
    assert from_mapping.total_free == 8
    assert from_mapping.total_occupied == 0
    assert list(from_mapping.bounds) == [None, None]

    from_object = standardize_availability_stats(
        SimpleNamespace(total_free=4, total_occupied=6, bounds=(0, 10))
    )
    assert from_object.total_space == 10
    assert from_object.free_chunks == 0
    with pytest.raises(ValueError, match="Cannot standardize stats"):
        standardize_availability_stats(object())

    performance = PerformanceStats(operation_count=4, cache_hits=1)
    assert performance.cache_hit_rate == 0.25
    assert standardize_performance_stats(performance) is performance
    mapped = standardize_performance_stats(
        {
            "operation_count": 5,
            "cache_hits": 2,
            "implementation": "mapped",
            "language": "python",
        }
    )
    assert mapped.cache_hit_rate == 0.4
    assert mapped.implementation_name == "mapped"
    shaped = standardize_performance_stats(
        SimpleNamespace(operation_count=3, cache_hits=1, implementation="object")
    )
    assert shaped.operation_count == 3
    assert shaped.language == ""
    with pytest.raises(ValueError, match="Cannot standardize performance stats"):
        standardize_performance_stats(object())


def test_backend_configuration_copies_mutable_compatibility_inputs() -> None:
    features = ["core"]
    constructor_args = {"seed": 1}
    config = BackendConfiguration(
        implementation_id="example",
        name="Example",
        language="python",
        implementation_type=ImplementationType.BOUNDARY,
        performance_tier=PerformanceTier.BASELINE,
        features=features,
        constructor_args=constructor_args,
    )
    features.append("late")
    constructor_args["seed"] = 2

    assert list(config.features) == ["core"]
    assert isinstance(config.constructor_args, MappingProxyType)
    assert config.constructor_args["seed"] == 1
    with pytest.raises(TypeError):
        config.constructor_args["seed"] = 3  # type: ignore[index]


def test_payload_policies_share_restriction_invariant_and_explicit_algebra() -> None:
    source = Span(0, 10)
    target = Span(2, 5)
    outside = Span(9, 12)
    copied = ["A"]
    uniform = UniformPayloadPolicy[list[str]](copy_on_split=True)
    restricted = uniform.restrict(copied, source, target)
    assert restricted == copied
    assert restricted is not copied
    assert uniform.can_merge(["A"], ["A"])
    with pytest.raises(ValueError, match="uniform payloads differ"):
        uniform.combine(["A"], ["B"])

    restricted_calls: list[tuple[Span, Span]] = []
    join = JoinPayloadPolicy[frozenset[str]](
        lambda left, right: left | right,
        frozenset(),
        restrict_fn=lambda data, old, new: restricted_calls.append((old, new)) or data,
    )
    assert join.combine(frozenset({"A"}), frozenset({"B"})) == frozenset({"A", "B"})
    assert join.restrict(frozenset({"A"}), source, target) == frozenset({"A"})
    assert len(restricted_calls) == 1
    assert restricted_calls[0][0] == source
    assert restricted_calls[0][1] == target

    ordered = OrderedPayloadPolicy[str](
        lambda left, right: left + right,
        "",
        restrict_fn=lambda data, old, new: f"{data}:{new.start}",
    )
    key = ordered.event_key("A")
    assert key[0] == "builtins"
    assert key[1] == "str"
    assert ordered.combine("A", "B") == "AB"
    assert ordered.restrict("A", source, target) == "A:2"
    custom = OrderedPayloadPolicy[str](
        lambda left, right: left + right,
        "",
        event_key_fn=str.lower,
    )
    assert custom.event_key("B") == "b"

    for policy, payload in (
        (uniform, ["A"]),
        (join, frozenset({"A"})),
        (ordered, "A"),
    ):
        with pytest.raises(ValueError, match="must be contained"):
            policy.restrict(payload, source, outside)  # type: ignore[arg-type]


def test_legacy_policy_covers_historical_merge_and_split_hooks() -> None:
    assert LegacyPayloadPolicy[str]().can_merge("A", "B")
    assert LegacyPayloadPolicy[str](
        can_merge_fn=lambda left, right: left == right
    ).can_merge("A", "A")
    assert LegacyPayloadPolicy[str](merge_idempotent=True).combine("A", "A") == "A"
    assert (
        LegacyPayloadPolicy[str](merge_fn=lambda left, right: left + right).combine(
            "A", "B"
        )
        == "AB"
    )
    assert LegacyPayloadPolicy[set[str]]().combine({"A"}, {"B"}) == {"A", "B"}
    assert LegacyPayloadPolicy[str]().combine("A", "B") == "A"

    calls: list[tuple[int, int, int, int]] = []
    policy = LegacyPayloadPolicy[str](
        split_fn=lambda data, old_start, old_end, new_start, new_end: (
            calls.append((old_start, old_end, new_start, new_end)) or data.lower()
        )
    )
    assert policy.restrict("A", Span(0, 10), Span(2, 5)) == "a"
    assert len(calls) == 1
    assert list(calls[0]) == [0, 10, 2, 5]
    with pytest.raises(ValueError, match="must be contained"):
        policy.restrict("A", Span(0, 10), Span(9, 12))


def test_interval_tree_base_delegates_to_canonical_legacy_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    split_calls: list[tuple[int, int]] = []
    tree = _Tree(
        merge_fn=lambda left, right: left + right,
        split_fn=lambda data, _old_start, _old_end, start, end: (
            split_calls.append((start, end)) or data.lower()
        ),
        can_merge=lambda left, right: left == right,
        merge_idempotent=True,
    )
    assert tree.merge_data(None, "A") == "A"
    assert tree.merge_data("A", None) == "A"
    assert tree.merge_data("A", "A") == "A"
    assert tree.merge_data("A", "B") == "AB"
    assert tree.split_data(None, 0, 10, 2, 5) is None
    assert tree.split_data("A", 0, 10, 2, 5) == "a"
    assert len(split_calls) == 1
    assert list(split_calls[0]) == [2, 5]
    assert tree.can_merge_data("A", "A")
    assert not tree.can_merge_data("A", "B")
    assert tree.get_total_available_length() == 0

    root = _Node(0, 2, "root")
    root.left = _Node(0, 1, "left")
    tree.root = root
    tree.print_tree()
    output = capsys.readouterr().out
    assert "0:2" in output
    assert "data: root" in output
    assert "data: left" in output


def test_public_rangeset_protocol_reexports_canonical_value_types() -> None:
    assert "add" in RangeSetProtocol.__dict__
    assert "allocate" in RangeSetProtocol.__dict__
    assert RangeSetProtocol.__module__ == "treemendous.protocols"
