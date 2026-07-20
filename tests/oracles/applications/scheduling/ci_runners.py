"""Independent CI label/concurrency reference."""


def expected_runner(
    runners: tuple[tuple[str, frozenset[str], int], ...], labels: frozenset[str]
) -> str | None:
    return min(
        (
            name
            for name, available, slots in runners
            if labels <= available and slots > 0
        ),
        default=None,
    )
