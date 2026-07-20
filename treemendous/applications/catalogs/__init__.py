"""Identity-preserving overlap and catalog application engines.

Import concrete engines from their scenario modules. Keeping this package
initializer data-free lets the central registry load ``manifest`` without
eagerly importing application implementations.
"""

__all__: list[str] = []
