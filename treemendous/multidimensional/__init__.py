"""Experimental multidimensional identity index; not part of the stable root API."""

from treemendous.multidimensional.diagnostics import BoxIndexDiagnostics
from treemendous.multidimensional.domain import (
    Box,
    BoxEntry,
    BoxHandle,
    BoxIndexSnapshot,
)
from treemendous.multidimensional.index import BoxIndex
from treemendous.multidimensional.protocols import BoxIndexProtocol

__all__ = [
    "Box",
    "BoxEntry",
    "BoxHandle",
    "BoxIndex",
    "BoxIndexDiagnostics",
    "BoxIndexProtocol",
    "BoxIndexSnapshot",
]
