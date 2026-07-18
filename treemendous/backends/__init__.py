"""Typed immutable backend catalog."""

from .catalog import CATALOG, CATALOG_BY_ID
from .probe import probe_backend
from .selection import select_backend
from .types import (
    Algorithm,
    Available,
    BackendDecision,
    BackendRequest,
    BackendSpec,
    Capability,
    Device,
    Invalid,
    Maturity,
    Runtime,
    Unavailable,
)

__all__ = [
    "Algorithm",
    "Available",
    "BackendDecision",
    "BackendRequest",
    "BackendSpec",
    "CATALOG",
    "CATALOG_BY_ID",
    "Capability",
    "Device",
    "Invalid",
    "Maturity",
    "Runtime",
    "Unavailable",
    "probe_backend",
    "select_backend",
]
