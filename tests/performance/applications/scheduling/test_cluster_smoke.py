from time import perf_counter

import pytest

from tests.oracles.applications.scheduling.cluster import expected_node
from tests.performance.applications.scheduling._shared import SmokeResult
from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications.scheduling.cluster import ClusterNode, ClusterScheduler


def run_smoke(operations: int = 64) -> SmokeResult:
    nodes = (
        ClusterNode("a", CapacityVector(cpu=4, memory=8), frozenset({"linux"})),
        ClusterNode("b", CapacityVector(cpu=4, memory=8), frozenset({"gpu"})),
    )
    scheduler = ClusterScheduler(nodes)
    reference = expected_node(
        tuple((n.name, n.capacity.to_dict(), n.labels) for n in nodes),
        {"cpu": 1, "memory": 2}, frozenset({"gpu"}),
    )
    started = perf_counter()
    for index in range(operations):
        placement = scheduler.schedule(
            f"job-{index}", 1, CapacityVector(cpu=1, memory=2),
            required_labels=frozenset({"gpu"}), earliest_start=index,
            latest_end=index + 1,
        )
        assert placement.resource == reference
    return SmokeResult(operations, operations, perf_counter() - started)


@pytest.mark.benchmark
def test_cluster_smoke_matches_oracle() -> None:
    assert run_smoke(16).oracle_checks == 16
