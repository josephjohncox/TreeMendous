"""Discovery and construction for canonical ``RangeSet`` backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from treemendous.domain import (
    BackendInvalidError,
    BackendUnavailableError,
    DomainInput,
)
from treemendous.policies import PayloadPolicy
from treemendous.rangeset import RangeSet

from .adapters import BackendAdapter
from .catalog import CATALOG
from .probe import probe_backend
from .selection import select_backend
from .types import (
    Available,
    BackendDecision,
    BackendRequest,
    BackendSpec,
    Invalid,
    ProbeState,
    Unavailable,
)


@dataclass(frozen=True)
class BackendRegistry:
    """Immutable discovered catalog with one construction interface."""

    specs: tuple[BackendSpec, ...]
    states: Mapping[str, ProbeState]

    def __post_init__(self) -> None:
        ids = tuple(spec.id for spec in self.specs)
        if len(set(ids)) != len(ids):
            raise ValueError("backend IDs must be unique")
        if set(self.states) != set(ids):
            raise ValueError("probe states must exactly match backend specs")
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))

    @classmethod
    def discover(
        cls,
        specs: tuple[BackendSpec, ...] = CATALOG,
        *,
        probe: Callable[[BackendSpec], ProbeState] = probe_backend,
    ) -> "BackendRegistry":
        return cls(specs, {spec.id: probe(spec) for spec in specs})

    def select(self, request: BackendRequest = BackendRequest()) -> BackendDecision:
        return select_backend(self.specs, self.states, request)

    def available_specs(self) -> tuple[BackendSpec, ...]:
        """Return semantically validated backend specifications in catalog order."""
        return tuple(
            spec for spec in self.specs if isinstance(self.states[spec.id], Available)
        )

    def create(
        self,
        domain: DomainInput,
        *,
        backend: str | None = None,
        request: BackendRequest = BackendRequest(),
        payload_policy: PayloadPolicy[Any] | None = None,
        payload_cloner: Callable[[Any], Any] = deepcopy,
        initially_available: bool = True,
    ) -> RangeSet:
        spec = self._resolve(backend, request)
        implementation = spec.loader()(**dict(spec.constructor_args))
        return RangeSet(
            BackendAdapter(implementation),
            domain=domain,
            payload_policy=payload_policy,
            payload_cloner=payload_cloner,
            initially_available=initially_available,
        )

    def _resolve(self, backend: str | None, request: BackendRequest) -> BackendSpec:
        if backend is None:
            return self.select(request).selected
        spec = next(
            (candidate for candidate in self.specs if candidate.id == backend), None
        )
        if spec is None:
            raise BackendUnavailableError(f"unknown backend: {backend}")
        state = self.states[backend]
        if isinstance(state, Invalid):
            raise BackendInvalidError(
                f"backend {backend} failed validation: {state.error}"
            )
        if isinstance(state, Unavailable):
            raise BackendUnavailableError(
                f"backend {backend} unavailable: {state.reason}"
            )
        assert isinstance(state, Available)
        return select_backend((spec,), {backend: state}, request).selected


@lru_cache(maxsize=1)
def _default_registry() -> BackendRegistry:
    return BackendRegistry.discover()


def create_range_set(
    domain: DomainInput,
    *,
    backend: str | None = None,
    request: BackendRequest = BackendRequest(),
    payload_policy: PayloadPolicy[Any] | None = None,
    payload_cloner: Callable[[Any], Any] = deepcopy,
    initially_available: bool = True,
) -> RangeSet:
    """Construct a canonical range set from the discovered default registry."""
    return _default_registry().create(
        domain,
        backend=backend,
        request=request,
        payload_policy=payload_policy,
        payload_cloner=payload_cloner,
        initially_available=initially_available,
    )
