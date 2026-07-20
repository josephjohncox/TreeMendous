"""Deterministic in-memory cluster capacity-vector scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import ReservationSnapshot
from treemendous.applications.scheduling._common import Placement
from treemendous.applications.scheduling._placement import (
    BoundedPlacementEngine,
    LabeledResource,
)


@dataclass(frozen=True)
class ClusterNode(LabeledResource):
    """A node whose capacity dimensions and labels constrain jobs."""


class ClusterScheduler:
    """Places jobs by earliest slot then node name; it is not a global optimizer."""

    def __init__(self, nodes: tuple[ClusterNode, ...]) -> None:
        self._engine = BoundedPlacementEngine(tuple(nodes))

    def schedule(
        self,
        job_id: str,
        duration: int,
        demand: CapacityVector,
        *,
        required_labels: frozenset[str] = frozenset(),
        earliest_start: int = 0,
        latest_end: int,
        request_id: str | None = None,
    ) -> Placement:
        return self._engine.place(
            job_id,
            duration,
            demand,
            required_labels=required_labels,
            earliest_start=earliest_start,
            latest_end=latest_end,
            request_id=request_id,
        )

    def cancel(self, job_id: str, reservation_id: str) -> Placement:
        return self._engine.cancel(job_id, reservation_id)

    def snapshot(self) -> ReservationSnapshot:
        return self._engine.snapshot()


def create_cluster_scheduler(
    *, nodes: tuple[ClusterNode, ...] | None = None
) -> ClusterScheduler:
    """Construct a cluster scheduler with an explicit or deterministic demo node."""
    if nodes is None:
        nodes = (
            ClusterNode(
                "node-a",
                CapacityVector(cpu=8, memory=32),
                frozenset({"linux"}),
            ),
        )
    return ClusterScheduler(nodes)
