from __future__ import annotations

import pytest

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.cluster import ClusterNode, ClusterScheduler


def test_cluster_capacity_labels_placement_and_failure_atomicity() -> None:
    scheduler = ClusterScheduler(
        (
            ClusterNode("a", CapacityVector(cpu=4, memory=8), frozenset({"linux"})),
            ClusterNode("b", CapacityVector(cpu=4, memory=8), frozenset({"linux", "gpu"})),
        )
    )
    demand = CapacityVector(cpu=3, memory=4)
    first = scheduler.schedule(
        "one", 2, demand, required_labels=frozenset({"gpu"}), latest_end=4,
        request_id="r1",
    )
    assert first.resource == "b"
    replay = scheduler.schedule(
        "one", 2, demand, required_labels=frozenset({"gpu"}), latest_end=4,
        request_id="r1",
    )
    assert replay is first
    before = scheduler.snapshot()
    with pytest.raises(SchedulingUnavailableError) as raised:
        scheduler.schedule(
            "two", 2, demand, required_labels=frozenset({"gpu"}), latest_end=2
        )
    assert len(raised.value.considered) == 1
    assert raised.value.considered[0] == "b"
    assert scheduler.snapshot() == before
    assert not scheduler.cancel("one", first.id).reservation.active
