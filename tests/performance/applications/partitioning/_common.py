"""Shared validation for bounded partitioning benchmark workloads."""


def validate_case(operations: int, seed: int, *, maximum: int) -> None:
    """Validate the uniform application benchmark arguments."""
    if isinstance(operations, bool) or not isinstance(operations, int):
        raise TypeError("operations must be an integer")
    if operations <= 0:
        raise ValueError("operations must be positive")
    if operations > maximum:
        raise ValueError(f"operations must not exceed {maximum}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
