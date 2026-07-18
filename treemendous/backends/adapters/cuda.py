"""CUDA remains experimental and unavailable to stable selection."""

from .base import CppBackendAdapter as CudaBackendAdapter

__all__ = ["CudaBackendAdapter"]
