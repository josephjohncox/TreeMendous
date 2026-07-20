"""Independent dependency-order validator for build sharding."""

from collections.abc import Mapping, Sequence


def is_dependency_order(order: Sequence[str], dependencies: Mapping[str, Sequence[str]]) -> bool:
    position = {name: index for index, name in enumerate(order)}
    return set(order) == set(dependencies) and all(position[dependency] < position[name] for name, values in dependencies.items() for dependency in values)
