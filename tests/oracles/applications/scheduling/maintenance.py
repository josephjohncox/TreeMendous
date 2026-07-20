"""Independent service-window/blackout/dependency reference."""


def valid_window(
    start: int,
    end: int,
    windows: tuple[tuple[int, int], ...],
    blackouts: tuple[tuple[int, int], ...],
    dependency_ready: int,
) -> bool:
    allowed = any(left <= start and end <= right for left, right in windows)
    blocked = any(left < end and start < right for left, right in blackouts)
    return dependency_ready <= start and allowed and not blocked
