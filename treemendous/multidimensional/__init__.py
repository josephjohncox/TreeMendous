"""Experimental multidimensional identity index; not part of the stable root API."""

from treemendous.multidimensional.diagnostics import BoxIndexDiagnostics
from treemendous.multidimensional.domain import (
    Box,
    BoxEntry,
    BoxHandle,
    BoxIndexSnapshot,
)
from treemendous.multidimensional.index import (
    BoundedBoxIndex,
    BoxIndex,
    BoxIndex2D,
    BoxIndex3D,
    BoxIndex4D,
)
from treemendous.multidimensional.protocols import BoxIndexProtocol

__all__ = [
    "BoundedBoxIndex",
    "Box",
    "BoxEntry",
    "BoxHandle",
    "BoxIndex",
    "BoxIndex2D",
    "BoxIndex3D",
    "BoxIndex4D",
    "BoxIndexDiagnostics",
    "BoxIndexProtocol",
    "BoxIndexSnapshot",
]
