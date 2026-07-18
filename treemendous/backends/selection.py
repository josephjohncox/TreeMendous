"""Pure capability-aware backend selection."""

from __future__ import annotations

from treemendous.domain import BackendUnavailableError

from .types import (
    Available,
    BackendDecision,
    BackendRequest,
    BackendSpec,
    Maturity,
    ProbeState,
)

_PRIORITY = (
    "cpp_boundary",
    # cpp_boundary_optimized is intentionally a parity alias, not a faster path.
    "cpp_boundary_optimized",
    "py_boundary_summary",
    "py_summary",
    "py_treap",
    "py_avl_earliest",
    "py_boundary",
)


def select_backend(
    specs: tuple[BackendSpec, ...],
    probes: dict[str, ProbeState],
    request: BackendRequest,
) -> BackendDecision:
    accepted: list[BackendSpec] = []
    rejected: list[tuple[str, str]] = []
    for spec in specs:
        probe = probes[spec.id]
        reason = None
        if spec.maturity is not Maturity.STABLE:
            reason = "experimental"
        elif not isinstance(probe, Available):
            reason = getattr(probe, "reason", getattr(probe, "error", "unavailable"))
        elif not request.require <= probe.validated_capabilities:
            reason = "missing required capability"
        elif spec.coordinate_bits < request.coordinate_bits:
            reason = "coordinate width too small"
        elif request.deterministic and not spec.deterministic:
            reason = "not deterministic"
        elif (
            request.preferred_runtime and spec.runtime is not request.preferred_runtime
        ):
            reason = "runtime does not match preference"
        if reason:
            rejected.append((spec.id, reason))
        else:
            accepted.append(spec)
    if not accepted:
        raise BackendUnavailableError("no backend satisfies the request")
    rank = {backend_id: index for index, backend_id in enumerate(_PRIORITY)}
    selected = min(accepted, key=lambda spec: rank.get(spec.id, len(rank)))
    return BackendDecision(selected, tuple(rejected))
