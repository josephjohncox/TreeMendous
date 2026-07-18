"""Strict semantic backend probing for the production catalog."""

from __future__ import annotations

from typing import Any

from treemendous.basic.protocols import (
    standardize_interval_result,
    standardize_intervals_list,
)

from .types import (
    Available,
    BackendSpec,
    Capability,
    Invalid,
    Maturity,
    ProbeState,
    Runtime,
    Unavailable,
)


def _shape(implementation: Any) -> tuple[tuple[int, int], ...]:
    return tuple(
        (item.start, item.end)
        for item in standardize_intervals_list(implementation.get_intervals())
    )


def _assert_invalid_atomic(implementation: Any, method: str) -> None:
    before = (_shape(implementation), implementation.get_total_available_length())
    try:
        getattr(implementation, method)(1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{method} must reject invalid spans with ValueError")
    after = (_shape(implementation), implementation.get_total_available_length())
    if after != before:
        raise AssertionError(f"failed {method} changed observable state")


def _probe_core(spec: BackendSpec, implementation: Any) -> None:
    if _shape(implementation) or implementation.get_total_available_length() != 0:
        raise AssertionError("new implementation is not empty")
    _assert_invalid_atomic(implementation, "release_interval")
    _assert_invalid_atomic(implementation, "reserve_interval")

    implementation.release_interval(0, 5)
    implementation.release_interval(10, 20)
    if _shape(implementation) != ((0, 5), (10, 20)):
        raise AssertionError("snapshot is not ordered and canonical")
    found = standardize_interval_result(implementation.find_interval(0, 8))
    if found is None or (found.start, found.end) != (10, 18):
        raise AssertionError("fragmented first-fit semantic check failed")
    containing = standardize_interval_result(implementation.find_interval(11, 3))
    if containing is None or (containing.start, containing.end) != (11, 14):
        raise AssertionError("containing-start first-fit semantic check failed")
    if standardize_interval_result(implementation.find_interval(0, 50)) is not None:
        raise AssertionError("no-fit query must return None")
    if implementation.get_total_available_length() != 15:
        raise AssertionError("total measure semantic check failed")

    implementation.reserve_interval(12, 15)
    if _shape(implementation) != ((0, 5), (10, 12), (15, 20)):
        raise AssertionError("reserve snapshot semantic check failed")
    if implementation.get_total_available_length() != 12:
        raise AssertionError("reserve total semantic check failed")

    if spec.runtime is Runtime.CPP:
        _probe_native_limits(spec)


def _probe_native_limits(spec: BackendSpec) -> None:
    implementation = spec.loader()(**dict(spec.constructor_args))
    minimum = -(2**63)
    maximum = 2**63 - 1
    implementation.release_interval(minimum, -1)
    before = (_shape(implementation), implementation.get_total_available_length())
    try:
        implementation.release_interval(-1, maximum)
    except OverflowError:
        pass
    else:
        raise AssertionError("aggregate overflow must raise OverflowError")
    if (_shape(implementation), implementation.get_total_available_length()) != before:
        raise AssertionError("overflowing release changed observable state")
    if implementation.find_interval(-10, maximum) is not None:
        raise AssertionError("huge negative-coordinate query escaped free coverage")
    for outside in (minimum - 1, maximum + 1):
        try:
            implementation.release_interval(outside, 0 if outside < 0 else outside + 1)
        except OverflowError:
            pass
        else:
            raise AssertionError("outside-int64 conversion must raise OverflowError")


def _rangeset(spec: BackendSpec, *, payload_policy: Any = None) -> Any:
    from treemendous.backends.adapters import CppBackendAdapter, PythonBackendAdapter
    from treemendous.rangeset import RangeSet

    implementation = spec.loader()(**dict(spec.constructor_args))
    adapter_type = (
        PythonBackendAdapter if spec.runtime is Runtime.PYTHON else CppBackendAdapter
    )
    return RangeSet(
        adapter_type(implementation),
        capabilities=spec.capabilities,
        initially_available=False,
        payload_policy=payload_policy,
    )


def _probe_payloads(spec: BackendSpec) -> None:
    from treemendous.domain import Span
    from treemendous.policies import (
        JoinPayloadPolicy,
        OrderedPayloadPolicy,
        UniformPayloadPolicy,
    )

    join_ranges = _rangeset(
        spec,
        payload_policy=JoinPayloadPolicy(lambda left, right: left | right, frozenset()),
    )
    join_ranges.add(Span(0, 5), frozenset({"A"}))
    join_ranges.add(Span(5, 10), frozenset({"B"}))
    fit = join_ranges.first_fit(
        10, not_before=0, payload_predicate=lambda data: bool(data)
    )
    if fit is None or fit.data != (frozenset({"A"}), frozenset({"B"})):
        raise AssertionError("payload query failed across accepted segments")

    uniform = _rangeset(spec, payload_policy=UniformPayloadPolicy())
    uniform.add(Span(0, 10), "A")
    before = uniform.snapshot()
    try:
        uniform.add(Span(5, 15), "B")
    except ValueError:
        pass
    else:
        raise AssertionError("uniform payload conflict was accepted")
    if uniform.snapshot() != before:
        raise AssertionError("uniform payload conflict changed state")

    policy = OrderedPayloadPolicy(
        lambda left, right: left + right, (), event_key_fn=lambda value: value
    )
    observed = []
    for events in (
        ((Span(0, 10), ("A",)), (Span(5, 15), ("B",))),
        ((Span(5, 15), ("B",)), (Span(0, 10), ("A",))),
    ):
        ordered = _rangeset(spec, payload_policy=policy)
        for span, data in events:
            ordered.add(span, data)
        observed.append(tuple((x.start, x.end, x.data) for x in ordered.intervals()))
    if observed[0] != observed[1]:
        raise AssertionError("ordered payload fold depends on insertion order")


def _probe_atomic_allocate(spec: BackendSpec) -> None:
    from treemendous.domain import Span

    ranges = _rangeset(spec)
    ranges.add(Span(0, 10))
    allocated = ranges.allocate(4, not_before=2, not_after=8)
    if allocated is None or allocated.span != Span(2, 6):
        raise AssertionError("atomic allocation returned the wrong fit")
    if tuple((x.start, x.end) for x in ranges.intervals()) != ((0, 2), (6, 10)):
        raise AssertionError("atomic allocation did not reserve exactly its result")


def _probe_analytics(spec: BackendSpec) -> None:
    implementation = spec.loader()(**dict(spec.constructor_args))
    if not hasattr(implementation, "get_availability_stats"):
        raise AssertionError("analytics API is absent")
    implementation.release_interval(0, 5)
    implementation.release_interval(10, 20)
    stats = implementation.get_availability_stats()
    if stats["total_free"] != 15 or stats["largest_chunk"] != 10:
        raise AssertionError("analytics semantic check failed")


def _probe_best_fit(spec: BackendSpec) -> None:
    implementation = spec.loader()(**dict(spec.constructor_args))
    if not hasattr(implementation, "find_best_fit"):
        raise AssertionError("best-fit API is absent")
    implementation.release_interval(0, 20)
    implementation.release_interval(30, 35)
    found = standardize_interval_result(implementation.find_best_fit(5, False))
    if found is None or (found.start, found.end) != (30, 35):
        raise AssertionError("best-fit semantic check failed")
    if standardize_interval_result(implementation.find_best_fit(50)) is not None:
        raise AssertionError("best-fit no-fit must return None")


def _probe_random_sample(spec: BackendSpec) -> None:
    implementation = spec.loader()(**dict(spec.constructor_args))
    if not hasattr(implementation, "sample_random_interval"):
        raise AssertionError("random-sampling API is absent")
    implementation.release_interval(3, 8)
    sampled = standardize_interval_result(implementation.sample_random_interval())
    if sampled is None or (sampled.start, sampled.end) != (3, 8):
        raise AssertionError("random sample escaped available geometry")


_CAPABILITY_PROBES = {
    Capability.PAYLOADS: _probe_payloads,
    Capability.ANALYTICS: _probe_analytics,
    Capability.BEST_FIT: _probe_best_fit,
    Capability.RANDOM_SAMPLE: _probe_random_sample,
    Capability.ATOMIC_ALLOCATE: _probe_atomic_allocate,
}


def _probe_capability(spec: BackendSpec, capability: Capability) -> None:
    probe = _CAPABILITY_PROBES.get(capability)
    if probe is None:
        raise AssertionError(f"no semantic probe for {capability.name}")
    probe(spec)


def probe_backend(spec: BackendSpec) -> ProbeState:
    if spec.maturity is Maturity.EXPERIMENTAL:
        return Unavailable("experimental backend is excluded from stable selection")
    try:
        implementation = spec.loader()(**dict(spec.constructor_args))
    except Exception as exc:
        return Unavailable(f"load failed: {exc}")
    try:
        if Capability.CORE in spec.capabilities:
            _probe_core(spec, implementation)
        extra_capabilities = spec.capabilities - {Capability.CORE}
        for capability in sorted(extra_capabilities, key=lambda item: item.value):
            _probe_capability(spec, capability)
    except Exception as exc:
        return Invalid(f"semantic check failed: {exc}")
    return Available(spec.capabilities)
