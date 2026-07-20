"""Independent fitness/ranking oracle for genetic populations."""

from collections.abc import Callable, Sequence


def expected_ranking(population: Sequence[str], fitness: Callable[[str], float]) -> tuple[tuple[float, str], ...]:
    return tuple(sorted(((fitness(item), item) for item in population), key=lambda item: (-item[0], item[1])))
