"""Independent all-or-nothing clinical-resource reference."""


def jointly_available(requested: tuple[str, ...], busy: frozenset[str]) -> bool:
    return all(resource not in busy for resource in requested)
