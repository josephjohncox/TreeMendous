"""Strict semantic backend probing for the production catalog."""

from __future__ import annotations

from typing import Any

from treemendous.backends.normalize import normalize_interval, normalize_intervals
from treemendous.domain import MutationResult

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
        for item in normalize_intervals(implementation.get_intervals())
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


def _delta_shape(delta: Any) -> tuple[tuple[int, int], ...]:
    return tuple((item.start, item.end) for item in normalize_intervals(delta.changed))


def _probe_authoritative_geometry(spec: BackendSpec) -> None:
    implementation = spec.loader()(**dict(spec.constructor_args))
    required = (
        "release_with_delta",
        "reserve_with_delta",
        "find_overlapping_intervals",
    )
    if not all(callable(getattr(implementation, name, None)) for name in required):
        return

    implementation.release_interval(0, 2)
    implementation.release_interval(4, 6)
    released = implementation.release_with_delta(1, 5)
    if not isinstance(released, MutationResult):
        raise AssertionError("authoritative mutations must return MutationResult")
    if (
        _delta_shape(released) != ((2, 4),)
        or released.changed_length != 2
        or released.fully_covered
        or _shape(implementation) != ((0, 6),)
    ):
        raise AssertionError("authoritative release delta is invalid")

    before = _shape(implementation)
    rejected = implementation.reserve_with_delta(1, 7, True)
    if not isinstance(rejected, MutationResult):
        raise AssertionError("authoritative mutations must return MutationResult")
    if (
        _delta_shape(rejected)
        or rejected.changed_length != 0
        or rejected.fully_covered
        or _shape(implementation) != before
    ):
        raise AssertionError("covered reservation rejection is not atomic")

    reserved = implementation.reserve_with_delta(1, 5, True)
    if not isinstance(reserved, MutationResult):
        raise AssertionError("authoritative mutations must return MutationResult")
    if (
        _delta_shape(reserved) != ((1, 5),)
        or reserved.changed_length != 4
        or not reserved.fully_covered
        or _shape(implementation) != ((0, 1), (5, 6))
    ):
        raise AssertionError("authoritative reserve delta is invalid")

    overlaps = normalize_intervals(implementation.find_overlapping_intervals(0, 6))
    if tuple((item.start, item.end) for item in overlaps) != (
        (0, 1),
        (5, 6),
    ):
        raise AssertionError("authoritative overlap query is invalid")

    if all(
        callable(getattr(implementation, name, None))
        for name in ("get_interval_count", "get_largest_available_length")
    ) and (
        implementation.get_interval_count() != 2
        or implementation.get_largest_available_length() != 1
    ):
        raise AssertionError("authoritative structural statistics are invalid")

    if callable(getattr(implementation, "allocate_interval", None)):
        implementation.release_interval(6, 12)
        allocated = normalize_interval(implementation.allocate_interval(5, 3, None))
        if (
            allocated is None
            or (allocated.start, allocated.end) != (5, 8)
            or _shape(implementation) != ((0, 1), (8, 12))
        ):
            raise AssertionError("authoritative allocation is not atomic")
        before = _shape(implementation)
        if (
            implementation.allocate_interval(8, 3, 10) is not None
            or _shape(implementation) != before
        ):
            raise AssertionError("bounded allocation rejection changed state")

    if callable(getattr(implementation, "set_managed_domain", None)):
        managed = spec.loader()(**dict(spec.constructor_args))
        managed.set_managed_domain([(0, 5), (10, 15)])
        managed.release_interval(0, 5)
        before = _shape(managed)
        try:
            managed.release_interval(5, 10)
        except ValueError:
            pass
        else:
            raise AssertionError("authoritative managed domain is not enforced")
        if _shape(managed) != before:
            raise AssertionError("managed-domain rejection changed state")


def _probe_core(spec: BackendSpec, implementation: Any) -> None:
    if _shape(implementation) or implementation.get_total_available_length() != 0:
        raise AssertionError("new implementation is not empty")
    _assert_invalid_atomic(implementation, "release_interval")
    _assert_invalid_atomic(implementation, "reserve_interval")

    implementation.release_interval(0, 5)
    implementation.release_interval(10, 20)
    if _shape(implementation) != ((0, 5), (10, 20)):
        raise AssertionError("snapshot is not ordered and canonical")
    found = normalize_interval(implementation.find_interval(0, 8))
    if found is None or (found.start, found.end) != (10, 18):
        raise AssertionError("fragmented first-fit semantic check failed")
    containing = normalize_interval(implementation.find_interval(11, 3))
    if containing is None or (containing.start, containing.end) != (11, 14):
        raise AssertionError("containing-start first-fit semantic check failed")
    if normalize_interval(implementation.find_interval(0, 50)) is not None:
        raise AssertionError("no-fit query must return None")
    if implementation.get_total_available_length() != 15:
        raise AssertionError("total measure semantic check failed")

    implementation.reserve_interval(12, 15)
    if _shape(implementation) != ((0, 5), (10, 12), (15, 20)):
        raise AssertionError("reserve snapshot semantic check failed")
    if implementation.get_total_available_length() != 12:
        raise AssertionError("reserve total semantic check failed")

    _probe_authoritative_geometry(spec)
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
    found = normalize_interval(implementation.find_best_fit(5, False))
    if found is None or (found.start, found.end) != (30, 35):
        raise AssertionError("best-fit semantic check failed")
    if normalize_interval(implementation.find_best_fit(50)) is not None:
        raise AssertionError("best-fit no-fit must return None")


def _probe_random_sample(spec: BackendSpec) -> None:
    implementation = spec.loader()(**dict(spec.constructor_args))
    if not hasattr(implementation, "sample_random_interval"):
        raise AssertionError("random-sampling API is absent")
    implementation.release_interval(3, 8)
    sampled = normalize_interval(implementation.sample_random_interval())
    if sampled is None or (sampled.start, sampled.end) != (3, 8):
        raise AssertionError("random sample escaped available geometry")


_CAPABILITY_PROBES = {
    Capability.ANALYTICS: _probe_analytics,
    Capability.BEST_FIT: _probe_best_fit,
    Capability.RANDOM_SAMPLE: _probe_random_sample,
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
