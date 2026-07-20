"""Pure capability-aware backend selection."""

from __future__ import annotations

from collections.abc import Mapping

from treemendous.domain import BackendUnavailableError

from .types import (
    Available,
    BackendDecision,
    BackendRequest,
    BackendSpec,
    Maturity,
    ProbeState,
    Unavailable,
)

_PRIORITY = (
    "cpp_boundary",
    "py_boundary_summary",
    "py_summary",
    "py_treap",
    "py_avl_earliest",
    "py_boundary",
)


def _rejection_reason(
    spec: BackendSpec, probe: ProbeState, request: BackendRequest
) -> str | None:
    """Return the first stable-selection invariant rejected by a request."""
    if spec.maturity is not Maturity.STABLE:
        return "experimental"
    if not isinstance(probe, Available):
        return getattr(probe, "reason", getattr(probe, "error", "unavailable"))
    if not request.require <= probe.validated_capabilities:
        return "missing required capability"
    if spec.coordinate_bits < request.coordinate_bits:
        return "coordinate width too small"
    if request.deterministic and not spec.deterministic:
        return "not deterministic"
    if request.preferred_runtime and spec.runtime is not request.preferred_runtime:
        return "runtime does not match preference"
    return None


def select_backend(
    specs: tuple[BackendSpec, ...],
    probes: Mapping[str, ProbeState],
    request: BackendRequest,
) -> BackendDecision:
    """Select deterministically from immutable specs and semantic probe results."""
    evaluated = tuple(
        (
            spec,
            _rejection_reason(
                spec,
                probes.get(spec.id, Unavailable("missing probe result")),
                request,
            ),
        )
        for spec in specs
    )
    accepted = tuple(spec for spec, reason in evaluated if reason is None)
    rejected = tuple((spec.id, reason) for spec, reason in evaluated if reason)
    if not accepted:
        raise BackendUnavailableError("no backend satisfies the request")
    rank = {backend_id: index for index, backend_id in enumerate(_PRIORITY)}
    selected = min(accepted, key=lambda spec: rank.get(spec.id, len(rank)))
    return BackendDecision(selected, rejected)
