"""Focused regressions for the canonical range-set contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from threading import Event, Thread

import pytest

import treemendous
from treemendous.backends import Available, CATALOG, Invalid, Maturity, probe_backend
from treemendous.backends.adapters import PythonBackendAdapter
from treemendous.basic.avl import IntervalNode, IntervalTree
from treemendous.basic.avl_earliest import EarliestIntervalTree
from treemendous.basic.boundary import IntervalManager
from treemendous.basic.boundary_summary import BoundarySummaryManager
from treemendous.basic.summary import SummaryIntervalTree
from treemendous.basic.treap import IntervalTreap, TreapNode
from treemendous.domain import ManagedDomain, Span, UnsupportedCapabilityError
from treemendous.policies import (
    JoinPayloadPolicy,
    OrderedPayloadPolicy,
    UniformPayloadPolicy,
)
from treemendous.rangeset import RangeSet


PYTHON_FACTORIES = [
    IntervalManager,
    EarliestIntervalTree,
    SummaryIntervalTree,
    lambda: IntervalTreap(random_seed=42),
    BoundarySummaryManager,
]


def _expect_equal(actual, expected):
    if actual != expected:
        pytest.fail(f"expected {expected!r}, got {actual!r}")


@pytest.mark.parametrize("factory", PYTHON_FACTORIES)
def test_fragmented_first_fit_is_complete(factory):
    manager = factory()
    manager.release_interval(0, 5)
    manager.release_interval(10, 20)
    result = manager.find_interval(0, 8)
    if not hasattr(result, "start"):
        assert result[0] == 10
        assert result[1] == 18
    else:
        assert result.start == 10
        assert result.end == 18


@pytest.mark.parametrize(
    "factory", [lambda: IntervalTree(IntervalNode), *PYTHON_FACTORIES]
)
@pytest.mark.parametrize("start,end", [(1, 1), (9, 4)])
def test_invalid_spans_are_atomic(factory, start, end):
    manager = factory()
    before = manager.get_total_available_length()
    with pytest.raises(ValueError):
        manager.release_interval(start, end)
    assert manager.get_total_available_length() == before


def test_atomic_allocate_and_explicit_domain_stats():
    ranges = RangeSet(
        PythonBackendAdapter(IntervalManager()),
        domain=ManagedDomain((0, 100)),
    )
    allocated = ranges.allocate(20, not_before=10, not_after=40)
    assert allocated is not None
    assert allocated.start == 10
    assert allocated.end == 30
    assert ranges.stats().total_occupied == 20
    assert ranges.allocate(80, not_before=0, not_after=80) is None
    assert ranges.snapshot().total_free == 80
    ranges.discard(Span(0, 10), require_covered=True)
    ranges.discard(Span(30, 100), require_covered=True)
    stats = ranges.stats()
    assert stats.total_free == 0
    assert stats.total_occupied == 100
    assert stats.utilization == 1.0


def test_treap_seed_is_forwarded_and_instance_local():
    first = treemendous.create_treap(1)
    second = treemendous.create_treap(2)
    first.release_interval(0, 10)
    second.release_interval(0, 10)
    assert (
        first.get_raw_implementation().root.priority
        != second.get_raw_implementation().root.priority
    )


def test_unordered_treap_merge_is_rejected_without_mutation():
    left = IntervalTreap(random_seed=1)
    right = IntervalTreap(random_seed=2)
    left.release_interval(50, 60)
    right.release_interval(10, 20)
    with pytest.raises(ValueError):
        left.merge_treap(right)
    left_intervals = left.get_intervals()
    right_intervals = right.get_intervals()
    assert len(left_intervals) == 1
    assert left_intervals[0].start == 50
    assert left_intervals[0].end == 60
    assert len(right_intervals) == 1
    assert right_intervals[0].start == 10
    assert right_intervals[0].end == 20


def test_backend_info_is_immutable():
    info = next(iter(treemendous.list_available_backends().values()))
    with pytest.raises(FrozenInstanceError):
        info.basic_ops_working = False


def test_adapter_does_not_retry_type_error():
    class MutatingFailure:
        calls = 0

        def release_interval(self, start, end, data=None):
            self.calls += 1
            raise TypeError("internal failure")

    raw = MutatingFailure()
    adapter = PythonBackendAdapter(raw)
    with pytest.raises(TypeError):
        adapter.release(0, 1)
    assert raw.calls == 1


def test_segment_tree_not_stable_export():
    import treemendous.basic as basic

    assert "SegmentTree" not in basic.__all__
    assert not hasattr(basic, "SegmentTree")


def test_mutation_result_reports_exact_changed_geometry():
    ranges = RangeSet(
        PythonBackendAdapter(IntervalManager()),
        domain=(0, 20),
        initially_available=True,
    )
    removed = ranges.discard(Span(5, 15))
    _expect_equal(removed.changed, (Span(5, 15),))
    assert removed.changed_length == 10
    assert removed.fully_covered

    partial = ranges.add(Span(10, 20))
    _expect_equal(partial.changed, (Span(10, 15),))
    assert partial.changed_length == 5
    assert not partial.fully_covered

    no_op = ranges.add(Span(10, 20))
    _expect_equal(no_op.changed, ())
    assert no_op.changed_length == 0
    assert no_op.fully_covered

    empty = RangeSet(PythonBackendAdapter(IntervalManager()), initially_available=False)
    absent = empty.add(Span(0, 10))
    assert not absent.fully_covered

    before = ranges.snapshot()
    rejected = ranges.discard(Span(0, 20), require_covered=True)
    _expect_equal(rejected.changed, ())
    assert not rejected.fully_covered
    assert ranges.snapshot() == before


def test_payload_policy_laws_and_endpoint_segmentation():
    join = JoinPayloadPolicy(lambda left, right: left | right, frozenset())
    a, b, c = (frozenset({value}) for value in "ABC")
    assert join.combine(join.combine(a, b), c) == join.combine(a, join.combine(b, c))
    assert join.combine(a, join.bottom) == a
    assert join.combine(a, a) == a

    ranges = RangeSet(
        PythonBackendAdapter(IntervalManager()),
        capabilities=frozenset(
            {treemendous.Capability.CORE, treemendous.Capability.PAYLOADS}
        ),
        initially_available=False,
        payload_policy=join,
    )
    ranges.add(Span(0, 10), a)
    ranges.add(Span(5, 15), b)
    _expect_equal(
        [(item.start, item.end, item.data) for item in ranges.intervals()],
        [(0, 5, a), (5, 10, a | b), (10, 15, b)],
    )


def test_uniform_and_ordered_payload_semantics_are_atomic_and_ordered():
    capabilities = frozenset(
        {treemendous.Capability.CORE, treemendous.Capability.PAYLOADS}
    )
    uniform = RangeSet(
        PythonBackendAdapter(IntervalManager()),
        capabilities=capabilities,
        initially_available=False,
        payload_policy=UniformPayloadPolicy(),
    )
    uniform.add(Span(0, 10), "A")
    before = uniform.snapshot()
    with pytest.raises(ValueError):
        uniform.add(Span(5, 15), "B")
    assert uniform.snapshot() == before
    uniform.add(Span(5, 15), "A")
    assert uniform.intervals()[0].span == Span(0, 15)

    ordered_policy = OrderedPayloadPolicy(lambda left, right: left + right, ())
    ordered = RangeSet(
        PythonBackendAdapter(IntervalManager()),
        capabilities=capabilities,
        initially_available=False,
        payload_policy=ordered_policy,
    )
    ordered.add(Span(0, 10), ("A",))
    ordered.add(Span(5, 15), ("B",))
    _expect_equal(
        [item.data for item in ordered.intervals()],
        [("A",), ("A", "B"), ("B",)],
    )
    _expect_equal(ordered_policy.combine(ordered_policy.identity, ("A",)), ("A",))


def test_ordered_payload_is_permutation_and_regrouping_invariant():
    capabilities = frozenset(
        {treemendous.Capability.CORE, treemendous.Capability.PAYLOADS}
    )
    policy = OrderedPayloadPolicy(
        lambda left, right: left + right,
        (),
        event_key_fn=lambda value: value,
    )
    snapshots = []
    events = ((Span(0, 10), ("A",)), (Span(5, 15), ("B",)), (Span(7, 12), ("C",)))
    for permutation in (
        events,
        tuple(reversed(events)),
        (events[1], events[0], events[2]),
    ):
        ranges = RangeSet(
            PythonBackendAdapter(IntervalManager()),
            capabilities=capabilities,
            initially_available=False,
            payload_policy=policy,
        )
        for span, data in permutation:
            ranges.add(span, data)
        # Splitting and restoring geometry must retain each event's stable key.
        ranges.discard(Span(8, 9))
        ranges.add(Span(8, 9), ("D",))
        snapshots.append(
            tuple((item.start, item.end, item.data) for item in ranges.intervals())
        )
    assert snapshots[0] == snapshots[1] == snapshots[2]
    a, b, c = ((value,) for value in "ABC")
    assert policy.combine(policy.combine(a, b), c) == policy.combine(
        a, policy.combine(b, c)
    )


def test_mutating_then_raising_payload_callback_is_atomic():
    def mutate_then_raise(left, right):
        left.append("MUTATED")
        raise RuntimeError("boom")

    ranges = RangeSet(
        PythonBackendAdapter(IntervalManager()),
        capabilities=frozenset(
            {treemendous.Capability.CORE, treemendous.Capability.PAYLOADS}
        ),
        initially_available=False,
        payload_policy=JoinPayloadPolicy(mutate_then_raise, []),
    )
    ranges.add(Span(0, 10), ["A"])
    before = ranges.snapshot()
    with pytest.raises(RuntimeError, match="boom"):
        ranges.add(Span(5, 15), ["B"])
    assert ranges.snapshot() == before
    assert ranges.intervals()[0].data == ["A"]


def test_payload_first_fit_spans_segments_and_returns_ordered_data():
    ranges = RangeSet(
        PythonBackendAdapter(IntervalManager()),
        capabilities=frozenset(
            {treemendous.Capability.CORE, treemendous.Capability.PAYLOADS}
        ),
        initially_available=False,
        payload_policy=JoinPayloadPolicy(lambda left, right: left | right, frozenset()),
    )
    ranges.add(Span(0, 5), frozenset({"A"}))
    ranges.add(Span(5, 10), frozenset({"B"}))
    expected = (frozenset({"A"}), frozenset({"B"}))
    fit = ranges.first_fit(
        10, not_before=0, payload_predicate=lambda value: bool(value)
    )
    assert fit is not None and fit.span == Span(0, 10) and fit.data == expected
    unfiltered = ranges.first_fit(10, not_before=0)
    assert unfiltered is not None and unfiltered.data == expected
    allocated = ranges.allocate(
        10, not_before=0, payload_predicate=lambda value: bool(value)
    )
    assert allocated is not None and allocated.data == expected
    _expect_equal(ranges.intervals(), ())


def test_public_create_range_set_options_are_not_backend_constructor_args():
    policy = JoinPayloadPolicy(lambda left, right: left | right, frozenset())
    ranges = treemendous.create_range_set(
        (0, 10),
        backend="py_boundary",
        require=frozenset(
            {treemendous.Capability.CORE, treemendous.Capability.PAYLOADS}
        ),
        initially_available=False,
        payload_policy=policy,
    )
    _expect_equal(ranges.intervals(), ())
    ranges.add(Span(0, 2), frozenset({"A"}))
    assert ranges.intervals()[0].data == frozenset({"A"})


def test_non_payload_adapter_rejects_payload_features():
    class GeometryOnly:
        def __init__(self):
            self.manager = IntervalManager()

        def release_interval(self, start, end):
            self.manager.release_interval(start, end)

        def reserve_interval(self, start, end):
            self.manager.reserve_interval(start, end)

        def find_interval(self, start, length):
            return self.manager.find_interval(start, length)

        def get_intervals(self):
            return self.manager.get_intervals()

        def get_total_available_length(self):
            return self.manager.get_total_available_length()

    from treemendous.backends.adapters import CppBackendAdapter

    ranges = RangeSet(CppBackendAdapter(GeometryOnly()), initially_available=False)
    with pytest.raises(UnsupportedCapabilityError):
        ranges.add(Span(0, 10), "lost")
    ranges.add(Span(0, 10))
    with pytest.raises(UnsupportedCapabilityError):
        ranges.first_fit(1, not_before=0, payload_predicate=lambda data: True)


def test_summary_domains_do_not_infer_convex_gaps_or_expand_on_reserve():
    for factory in (SummaryIntervalTree, BoundarySummaryManager):
        manager = factory()
        before = manager.get_availability_stats()
        manager.reserve_interval(100, 200)
        assert manager.get_availability_stats() == before
        manager.release_interval(0, 10)
        manager.release_interval(20, 30)
        stats = manager.get_availability_stats()
        assert stats["total_space"] == 20
        assert stats["total_free"] == 20
        assert stats["total_occupied"] == 0
        manager.reserve_interval(0, 10)
        manager.reserve_interval(20, 30)
        full = manager.get_availability_stats()
        assert full["total_free"] == 0
        assert full["total_occupied"] == 20
        assert full["utilization"] == 1.0


def test_treap_verifier_propagates_endpoint_bounds_and_merge_allows_adjacency():
    treap = IntervalTreap(random_seed=1)
    root = TreapNode(50, 60, priority=1.0)
    root.left = TreapNode(30, 35, priority=0.9)
    root.left.right = TreapNode(40, 55, priority=0.8)
    root.left.right.update_stats()
    root.left.update_stats()
    root.update_stats()
    treap.root = root
    assert not treap.verify_treap_properties()

    left = IntervalTreap(random_seed=7)
    right = IntervalTreap(random_seed=8)
    left.release_interval(0, 10)
    right.release_interval(10, 20)
    merged = left.merge_treap(right)
    _expect_equal(
        [(item.start, item.end) for item in merged.get_intervals()],
        [(0, 10), (10, 20)],
    )
    assert merged.verify_treap_properties()


def test_legacy_mutations_share_allocate_lock_and_return_none():
    manager = treemendous.create_interval_tree("py_boundary")
    assert manager.release_interval(0, 100) is None
    entered = Event()
    proceed = Event()
    mutation_done = Event()
    original = manager.first_fit

    def paused_first_fit(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert proceed.wait(timeout=2)
        return result

    manager.first_fit = paused_first_fit
    allocation = Thread(target=lambda: manager.allocate(10, not_before=0))
    mutation = Thread(
        target=lambda: (
            manager.release_interval(200, 210),
            mutation_done.set(),
        )
    )
    allocation.start()
    assert entered.wait(timeout=2)
    mutation.start()
    assert not mutation_done.wait(timeout=0.05)
    proceed.set()
    allocation.join(timeout=2)
    mutation.join(timeout=2)
    assert mutation_done.is_set()


def test_adapter_backed_reads_share_mutation_lock():
    class PausingManager(IntervalManager):
        def __init__(self):
            super().__init__()
            self.entered = Event()
            self.proceed = Event()

        def release_interval(self, start, end, data=None):
            self.entered.set()
            assert self.proceed.wait(timeout=2)
            return super().release_interval(start, end, data)

    raw = PausingManager()
    ranges = RangeSet(
        PythonBackendAdapter(raw),
        domain=(0, 20),
        initially_available=False,
    )
    add_done = Event()
    stats_done = Event()
    total_done = Event()
    mutation = Thread(target=lambda: (ranges.add(Span(0, 10)), add_done.set()))
    stats_reader = Thread(target=lambda: (ranges.stats(), stats_done.set()))
    total_reader = Thread(
        target=lambda: (ranges.get_total_available_length(), total_done.set())
    )
    mutation.start()
    assert raw.entered.wait(timeout=2)
    stats_reader.start()
    total_reader.start()
    assert not stats_done.wait(timeout=0.05)
    assert not total_done.wait(timeout=0.05)
    raw.proceed.set()
    for thread in (mutation, stats_reader, total_reader):
        thread.join(timeout=2)
    assert add_done.is_set() and stats_done.is_set() and total_done.is_set()


def test_false_declared_capability_is_invalid():
    boundary = next(spec for spec in CATALOG if spec.id == "py_boundary")
    false_claim = replace(
        boundary,
        id="synthetic_false_random",
        capabilities=boundary.capabilities | {treemendous.Capability.RANDOM_SAMPLE},
    )
    state = probe_backend(false_claim)
    assert isinstance(state, Invalid)
    if "random-sampling API is absent" not in state.error:
        pytest.fail(f"unexpected probe error: {state.error}")


def test_every_stable_catalog_entry_has_a_valid_probe():
    for spec in CATALOG:
        if spec.maturity is Maturity.STABLE:
            state = probe_backend(spec)
            if isinstance(state, Available):
                assert state.validated_capabilities == spec.capabilities
            else:
                # Missing optional extension modules are platform availability,
                # never semantic invalidity. This local suite has CPU extensions.
                assert state.__class__.__name__ == "Unavailable"
