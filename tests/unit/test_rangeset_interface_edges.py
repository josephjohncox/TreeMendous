from __future__ import annotations

import pytest

from treemendous.backends.adapters import CppBackendAdapter, PythonBackendAdapter
from treemendous.backends.types import Capability
from treemendous.basic.boundary import IntervalManager
from treemendous.domain import (
    ManagedDomainRequiredError,
    Span,
    UnsupportedCapabilityError,
)
from treemendous.policies import OrderedPayloadPolicy, UniformPayloadPolicy
from treemendous.rangeset import RangeSet

CORE = frozenset({Capability.CORE})
PAYLOADS = frozenset({Capability.CORE, Capability.PAYLOADS})


def _python_ranges(**options) -> RangeSet:
    return RangeSet(PythonBackendAdapter(IntervalManager()), **options)


def test_constructor_enforces_payload_capability_at_the_adapter_seam() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="adapter cannot preserve"):
        RangeSet(
            CppBackendAdapter(IntervalManager()),
            capabilities=PAYLOADS,
            initially_available=False,
        )
    with pytest.raises(UnsupportedCapabilityError, match="payload policies"):
        _python_ranges(
            capabilities=CORE,
            payload_policy=UniformPayloadPolicy(),
            initially_available=False,
        )


def test_initial_payload_state_uses_each_policy_identity() -> None:
    uniform = _python_ranges(domain=(0, 4), capabilities=PAYLOADS)
    assert uniform.domain is not None
    assert list(uniform.domain.bounds) == [0, 4]
    assert uniform.capabilities == PAYLOADS
    assert isinstance(uniform.payload_policy, UniformPayloadPolicy)
    assert uniform.intervals()[0].data is None

    policy = OrderedPayloadPolicy[tuple[str, ...]](
        lambda left, right: left + right,
        tuple(),
        event_key_fn=lambda value: value,
    )
    ordered = _python_ranges(
        domain=(0, 4),
        capabilities=PAYLOADS,
        payload_policy=policy,
    )
    assert ordered.payload_policy is policy
    assert ordered.intervals()[0].data == tuple()


def test_rangeset_edge_paths_remain_failure_atomic_and_observable() -> None:
    ranges = _python_ranges(
        domain=(0, 10),
        capabilities=PAYLOADS,
        initially_available=False,
    )
    ranges.release_interval(1, 5, data="A")
    before = ranges.snapshot()

    with pytest.raises(ValueError, match="contained in the managed domain"):
        ranges.add(Span(9, 12), "A")
    with pytest.raises(ValueError, match="not_after"):
        ranges.first_fit(1, not_before=4, not_after=4)
    assert ranges.snapshot() == before

    mutation = ranges.discard(Span(0, 2), require_covered=True)
    assert not mutation.changed
    assert mutation.changed_length == 0
    assert not mutation.fully_covered
    assert ranges.allocate(20, not_before=0) is None

    overlaps = ranges.overlaps(Span(3, 7))
    assert len(overlaps) == 1
    assert overlaps[0].span == Span(1, 5)
    assert ranges.find_interval(2, 2) is not None
    assert ranges.get_intervals() == list(ranges.intervals())
    assert ranges.get_total_available_length() == 4
    assert ranges.get_availability_stats() == ranges.stats()
    assert isinstance(ranges.get_raw_implementation(), IntervalManager)
    ranges.require(Capability.PAYLOADS)
    with pytest.raises(UnsupportedCapabilityError, match="ANALYTICS"):
        ranges.require(Capability.ANALYTICS)


def test_stats_require_an_explicit_domain() -> None:
    ranges = _python_ranges(capabilities=CORE, initially_available=False)
    with pytest.raises(ManagedDomainRequiredError, match="explicit managed domain"):
        ranges.stats()
