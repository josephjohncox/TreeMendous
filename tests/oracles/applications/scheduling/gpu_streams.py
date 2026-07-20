"""Independent GPU compatibility reference."""


def expected_pair(
    devices: tuple[tuple[str, frozenset[str], tuple[tuple[str, frozenset[str]], ...]], ...],
    compatibility: str,
) -> tuple[str, str] | None:
    pairs = [
        (device, stream)
        for device, device_kinds, streams in devices
        if compatibility in device_kinds
        for stream, stream_kinds in streams
        if compatibility in stream_kinds
    ]
    return min(pairs, default=None)
