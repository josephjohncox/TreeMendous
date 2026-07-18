"""Immutable backend catalog value types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class Capability(Enum):
    CORE = auto()
    PAYLOADS = auto()
    ANALYTICS = auto()
    BEST_FIT = auto()
    RANDOM_SAMPLE = auto()
    ATOMIC_ALLOCATE = auto()


class Algorithm(Enum):
    BOUNDARY = "boundary"
    AVL = "avl"
    SUMMARY = "summary"
    TREAP = "treap"


class Runtime(Enum):
    PYTHON = "python"
    CPP = "cpp"
    CUDA = "cuda"
    METAL = "metal"


class Device(Enum):
    CPU = "cpu"
    GPU = "gpu"


class Maturity(Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"


BackendId = str


@dataclass(frozen=True)
class BackendSpec:
    id: BackendId
    name: str
    algorithm: Algorithm
    runtime: Runtime
    device: Device
    maturity: Maturity
    capabilities: frozenset[Capability]
    coordinate_bits: int
    deterministic: bool
    loader: Callable[[], type[Any]]
    constructor_args: Mapping[str, Any]


@dataclass(frozen=True)
class Available:
    validated_capabilities: frozenset[Capability]


@dataclass(frozen=True)
class Unavailable:
    reason: str


@dataclass(frozen=True)
class Invalid:
    error: str


ProbeState = Available | Unavailable | Invalid


@dataclass(frozen=True)
class BackendRequest:
    require: frozenset[Capability] = frozenset({Capability.CORE})
    coordinate_bits: int = 64
    deterministic: bool = True
    preferred_runtime: Runtime | None = None


@dataclass(frozen=True)
class BackendDecision:
    selected: BackendSpec
    rejected: tuple[tuple[BackendId, str], ...]
