"""Shared correctness evidence for deterministic scheduling benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any, Literal, TypeVar

from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from treemendous.applications._shared.reservations import (
    Reservation,
    ReservationSnapshot,
)
from treemendous.applications.scheduling._common import Placement

DEFAULT_OPERATIONS = 64
DEFAULT_SEED = 20_260_720
MAX_OPERATIONS = 2_048

ResultT = TypeVar("ResultT")
SchedulerT = TypeVar("SchedulerT")


@dataclass(frozen=True)
class PlanStep:
    """One deterministic reserve or cancellation slot in a bounded workload."""

    action: Literal["reserve", "cancel"]
    ordinal: int
    variant: int


@dataclass(frozen=True)
class SchedulingCommand:
    """A fully prepared public scheduler invocation."""

    action: Literal["reserve", "cancel"]
    owner: str
    arguments: Mapping[str, Any]
    reservation_id: str | None = None


def make_plan(operations: int, seed: int) -> tuple[PlanStep, ...]:
    """Build a deterministic bounded mix with every fourth call cancelling."""
    if isinstance(operations, bool) or not isinstance(operations, int):
        raise TypeError("operations must be an integer")
    if not 0 < operations <= MAX_OPERATIONS:
        raise ValueError(f"operations must be between 1 and {MAX_OPERATIONS}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    random = Random(seed)
    active: list[int] = []
    schedule_count = 0
    steps: list[PlanStep] = []
    for operation in range(operations):
        if operation % 4 == 3 and active:
            position = random.randrange(len(active))
            ordinal = active.pop(position)
            steps.append(PlanStep("cancel", ordinal, 0))
        else:
            ordinal = schedule_count
            schedule_count += 1
            active.append(ordinal)
            steps.append(PlanStep("reserve", ordinal, random.getrandbits(32)))
    return tuple(steps)


def reservation_evidence(reservation: Reservation) -> dict[str, Any]:
    """Detach every public field of a reservation into primitive evidence."""
    return {
        "id": reservation.id,
        "owner": reservation.owner,
        "start": reservation.start,
        "end": reservation.end,
        "requirements": tuple(
            (item.resource, item.capacity.to_dict())
            for item in reservation.requirements
        ),
        "buffer_before": reservation.buffer_before,
        "buffer_after": reservation.buffer_after,
        "request_id": reservation.request_id,
        "status": reservation.status.value,
    }


def expected_reservation(
    *,
    owner: str,
    start: int,
    end: int,
    requirements: Sequence[tuple[str, Mapping[str, int]]],
    request_id: str | None,
    status: str = "active",
    buffer_before: int = 0,
    buffer_after: int = 0,
) -> dict[str, Any]:
    """Construct independent primitive reservation evidence."""
    return {
        "id": f"{owner}:1",
        "owner": owner,
        "start": start,
        "end": end,
        "requirements": tuple(
            (resource, dict(capacity)) for resource, capacity in sorted(requirements)
        ),
        "buffer_before": buffer_before,
        "buffer_after": buffer_after,
        "request_id": request_id,
        "status": status,
    }


def placement_evidence(placement: Placement) -> dict[str, Any]:
    """Detach a placement and its complete reservation."""
    return {
        "resource": placement.resource,
        "reservation": reservation_evidence(placement.reservation),
    }


def snapshot_evidence(snapshot: ReservationSnapshot) -> dict[str, Any]:
    """Detach a complete public reservation snapshot."""
    return {
        "resources": tuple(
            (item.resource, item.capacity.to_dict()) for item in snapshot.resources
        ),
        "reservations": tuple(
            reservation_evidence(item) for item in snapshot.reservations
        ),
    }


def expected_snapshot(
    resources: Mapping[str, Mapping[str, int]],
    reservations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Construct independently sorted primitive snapshot evidence."""
    return {
        "resources": tuple(
            (resource, dict(capacity))
            for resource, capacity in sorted(resources.items())
        ),
        "reservations": tuple(
            sorted(
                reservations,
                key=lambda item: (item["start"], item["end"], item["id"]),
            )
        ),
    }


def scheduling_counters(
    *, operations: int, results: Sequence[Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    """Account for operation results, lifecycle state, and owner ID counters."""
    reservations = state["reservations"]
    active = tuple(item for item in reservations if item["status"] == "active")
    return {
        "operations": operations,
        "results": len(results),
        "reservations": len(reservations),
        "active": len(active),
        "cancelled": len(reservations) - len(active),
        "service_slots": sum(item["end"] - item["start"] for item in reservations),
        "active_occupied_slots": sum(
            item["end"] + item["buffer_after"] - item["start"] + item["buffer_before"]
            for item in active
        ),
        "resource_claims": sum(len(item["requirements"]) for item in reservations),
        "next_owner_sequences": tuple(
            sorted((item["owner"], 2) for item in reservations)
        ),
    }


def reservation_oracle_outcome(
    *,
    operations: int,
    commands: Sequence[SchedulingCommand],
    resources: Mapping[str, Mapping[str, int]],
    reserve_record: Callable[[SchedulingCommand], dict[str, Any]],
    result_record: Callable[[SchedulingCommand, dict[str, Any]], Any],
) -> ApplicationOutcome:
    """Run an independent primitive lifecycle model after measured execution."""
    records: dict[str, dict[str, Any]] = {}
    results: list[Any] = []
    for command in commands:
        if command.action == "reserve":
            record = reserve_record(command)
        else:
            record = {**records[command.owner], "status": "cancelled"}
        records[command.owner] = record
        results.append(result_record(command, record))
    state = expected_snapshot(resources, tuple(records.values()))
    result_tuple = tuple(results)
    return ApplicationOutcome(
        result_tuple,
        state,
        scheduling_counters(operations=operations, results=result_tuple, state=state),
    )


def run_reservation_case(
    *,
    scenario_id: str,
    operations: int,
    scheduler: SchedulerT,
    commands: Sequence[SchedulingCommand],
    invoke: Callable[[SchedulerT, SchedulingCommand], ResultT],
    result_evidence: Callable[[ResultT], Any],
    oracle: Callable[[], ApplicationOutcome],
) -> ApplicationSample:
    """Time only prepared public calls, then attest the exact scheduler instance."""

    def execute() -> list[ResultT]:
        results: list[ResultT] = []
        for command in commands:
            results.append(invoke(scheduler, command))
        return results

    def observe(raw: list[ResultT]) -> ApplicationOutcome:
        results = tuple(result_evidence(item) for item in raw)
        state = snapshot_evidence(scheduler.snapshot())  # type: ignore[attr-defined]
        counters = scheduling_counters(
            operations=operations, results=results, state=state
        )
        return ApplicationOutcome(results, state, counters)

    return run_application_case(
        scenario_id=scenario_id,
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )
