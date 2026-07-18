"""Metal remains experimental and uses the non-payload native adapter."""

from .base import CppBackendAdapter as MetalBackendAdapter

__all__ = ["MetalBackendAdapter"]
