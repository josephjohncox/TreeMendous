"""Explicit payload algebra for canonical range operations."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from treemendous.domain import Span

D = TypeVar("D")


def _require_subspan(source: Span, target: Span) -> None:
    if not source.contains(target):
        raise ValueError("restriction target must be contained in source")


class PayloadPolicy(Protocol[D]):
    """Policy governing payload combination and subspan restriction."""

    def can_merge(self, left: D, right: D) -> bool: ...
    def combine(self, left: D, right: D) -> D: ...
    def restrict(self, data: D, source: Span, target: Span) -> D: ...


@dataclass(frozen=True)
class UniformPayloadPolicy(Generic[D]):
    """Uniform labels merge only when equal; splits copy the label."""

    copy_on_split: bool = False

    def can_merge(self, left: D, right: D) -> bool:
        return left == right

    def combine(self, left: D, right: D) -> D:
        if left != right:
            raise ValueError("uniform payloads differ")
        return left

    def restrict(self, data: D, source: Span, target: Span) -> D:
        _require_subspan(source, target)
        return copy(data) if self.copy_on_split else data


@dataclass(frozen=True)
class JoinPayloadPolicy(Generic[D]):
    """Pointwise commutative-idempotent join policy.

    Algebraic laws are caller obligations and are exercised by the supplied
    law-test helpers; Python's type system cannot prove them.
    """

    join: Callable[[D, D], D]
    bottom: D
    restrict_fn: Callable[[D, Span, Span], D] | None = None

    def can_merge(self, left: D, right: D) -> bool:
        return left == right

    def combine(self, left: D, right: D) -> D:
        return self.join(left, right)

    def restrict(self, data: D, source: Span, target: Span) -> D:
        _require_subspan(source, target)
        return self.restrict_fn(data, source, target) if self.restrict_fn else data


@dataclass(frozen=True)
class OrderedPayloadPolicy(Generic[D]):
    """Associative payload fold in stable coordinate/event-key order.

    Active events are ordered by their original ``(start, end)`` coordinates
    and then by ``event_key_fn(payload)``.  Supplying ``event_key_fn`` is
    recommended for application values; the default key is a stable type/repr
    key.  The key is retained when an event is split, so insertion permutation
    and later endpoint regrouping cannot change fold order.
    """

    combine_fn: Callable[[D, D], D]
    identity: D
    restrict_fn: Callable[[D, Span, Span], D] | None = None
    event_key_fn: Callable[[D], Any] | None = None

    def event_key(self, data: D) -> Any:
        if self.event_key_fn is not None:
            return self.event_key_fn(data)
        value_type = type(data)
        return (value_type.__module__, value_type.__qualname__, repr(data))

    def can_merge(self, left: D, right: D) -> bool:
        # Adjacent regions may coalesce only when their pointwise values agree;
        # folding distinct neighbours would smear events across an endpoint.
        return left == right

    def combine(self, left: D, right: D) -> D:
        return self.combine_fn(left, right)

    def restrict(self, data: D, source: Span, target: Span) -> D:
        _require_subspan(source, target)
        return self.restrict_fn(data, source, target) if self.restrict_fn else data
