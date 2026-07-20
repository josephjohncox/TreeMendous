"""CI runner reservations with labels, concurrency, and cancellation."""

from __future__ import annotations

from dataclasses import dataclass

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import ReservationSnapshot
from treemendous.applications.scheduling._common import Placement, integer, names, text
from treemendous.applications.scheduling._placement import (
    BoundedPlacementEngine,
    LabeledResource,
)


@dataclass(frozen=True)
class Runner:
    """A CI runner with exact-match labels and positive concurrent job slots."""

    name: str
    labels: frozenset[str]
    concurrency: int = 1

    def __post_init__(self) -> None:
        text(self.name, "runner name")
        names(self.labels, "runner labels")
        integer(self.concurrency, "concurrency", minimum=1)


class CIRunnerScheduler:
    """Selects the earliest compatible runner, breaking ties by runner name."""

    def __init__(self, runners: tuple[Runner, ...]) -> None:
        resources = tuple(
            LabeledResource(
                runner.name,
                CapacityVector(slots=runner.concurrency),
                runner.labels,
            )
            for runner in runners
        )
        self._engine = BoundedPlacementEngine(resources)

    def schedule(
        self,
        job_id: str,
        duration: int,
        *,
        labels: frozenset[str] = frozenset(),
        release_time: int = 0,
        deadline: int,
        request_id: str | None = None,
    ) -> Placement:
        return self._engine.place(
            job_id,
            duration,
            CapacityVector(slots=1),
            required_labels=labels,
            earliest_start=release_time,
            latest_end=deadline,
            request_id=request_id,
        )

    def cancel(self, job_id: str, reservation_id: str) -> Placement:
        return self._engine.cancel(job_id, reservation_id)

    def snapshot(self) -> ReservationSnapshot:
        return self._engine.snapshot()


def create_ci_runner_scheduler(
    *, runners: tuple[Runner, ...] | None = None
) -> CIRunnerScheduler:
    """Construct a CI scheduler."""
    return CIRunnerScheduler(
        runners or (Runner("runner-a", frozenset({"linux", "x86"}), concurrency=2),)
    )
