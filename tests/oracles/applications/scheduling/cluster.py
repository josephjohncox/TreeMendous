"""Independent cluster placement reference."""


def expected_node(
    nodes: tuple[tuple[str, dict[str, int], frozenset[str]], ...],
    demand: dict[str, int],
    labels: frozenset[str],
) -> str | None:
    eligible = []
    for name, capacity, node_labels in nodes:
        if labels <= node_labels and capacity.keys() == demand.keys() and all(
            demand[key] <= capacity[key] for key in demand
        ):
            eligible.append(name)
    return min(eligible, default=None)
