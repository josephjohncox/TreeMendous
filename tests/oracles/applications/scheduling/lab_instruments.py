"""Independent calibration and cleanup reference."""


def calibrated(
    start: int, end: int, cleanup: int, windows: tuple[tuple[int, int], ...]
) -> bool:
    return any(left <= start and end + cleanup <= right for left, right in windows)
