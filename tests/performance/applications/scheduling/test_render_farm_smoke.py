from typing import Any

import pytest

from tests.performance.applications.harness import (
    ApplicationOutcome,
    ApplicationSample,
    run_application_case,
)
from tests.performance.applications.scheduling._shared import (
    DEFAULT_OPERATIONS,
    DEFAULT_SEED,
    SchedulingCommand,
    expected_reservation,
    expected_snapshot,
    make_plan,
    placement_evidence,
    snapshot_evidence,
)
from treemendous.applications.scheduling.render_farm import (
    FrameChunkAssignment,
    RenderFarmScheduler,
    RenderFarmSnapshot,
    RenderWorker,
)


def _assignment_evidence(assignment: FrameChunkAssignment):
    return {
        "render_id": assignment.render_id,
        "frames": (assignment.frames.start, assignment.frames.end),
        "attempt": assignment.attempt,
        "status": assignment.status.value,
        "placement": placement_evidence(assignment.placement),
    }


def _snapshot_evidence(snapshot: RenderFarmSnapshot):
    return {
        "assignments": tuple(
            _assignment_evidence(item) for item in snapshot.assignments
        ),
        "reservations": snapshot_evidence(snapshot.reservations),
    }


def _counters(operations: int, results, state):
    assignments = state["assignments"]
    reservations = state["reservations"]["reservations"]
    return {
        "operations": operations,
        "results": len(results),
        "assignments": len(assignments),
        "active_assignments": sum(item["status"] == "active" for item in assignments),
        "cancelled_assignments": sum(
            item["status"] == "cancelled" for item in assignments
        ),
        "retried_assignments": sum(item["status"] == "retried" for item in assignments),
        "frames": sum(item["frames"][1] - item["frames"][0] for item in assignments),
        "reservations": len(reservations),
        "active_reservations": sum(item["status"] == "active" for item in reservations),
        "next_owner_sequences": tuple(
            sorted((item["owner"], 2) for item in reservations)
        ),
    }


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    scheduler = RenderFarmScheduler(
        (RenderWorker("worker-a"), RenderWorker("worker-b"))
    )
    commands: list[SchedulingCommand] = []
    for step in make_plan(operations, seed):
        owner = f"render-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}, f"{owner}:1"))
            continue
        start = step.ordinal * 3 + step.variant % 2
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "frame_start": step.ordinal * 10 + step.variant % 3,
                    "frame_count": 5,
                    "duration": 1,
                    "earliest_start": start,
                    "latest_end": start + 1,
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def execute() -> list[FrameChunkAssignment]:
        results = []
        for command in prepared:
            if command.action == "cancel":
                results.append(scheduler.cancel(command.owner, command.reservation_id))  # type: ignore[arg-type]
            else:
                results.append(
                    scheduler.assign_chunk(command.owner, **command.arguments)
                )
        return results

    def observe(raw: list[FrameChunkAssignment]) -> ApplicationOutcome:
        results = tuple(_assignment_evidence(item) for item in raw)
        state = _snapshot_evidence(scheduler.snapshot())
        return ApplicationOutcome(results, state, _counters(operations, results, state))

    def oracle() -> ApplicationOutcome:
        assignments: dict[str, dict[str, Any]] = {}
        reservations: dict[str, dict[str, Any]] = {}
        results: list[dict[str, Any]] = []
        for command in prepared:
            if command.action == "reserve":
                arguments = command.arguments
                reservation = expected_reservation(
                    owner=command.owner,
                    start=arguments["earliest_start"],
                    end=arguments["latest_end"],
                    requirements=(("worker-a", {"slots": 1}),),
                    request_id=arguments["request_id"],
                )
                assignment = {
                    "render_id": command.owner,
                    "frames": (
                        arguments["frame_start"],
                        arguments["frame_start"] + arguments["frame_count"],
                    ),
                    "attempt": 1,
                    "status": "active",
                    "placement": {
                        "resource": "worker-a",
                        "reservation": reservation,
                    },
                }
                reservations[command.owner] = reservation
            else:
                cancelled_reservation = {
                    **reservations[command.owner],
                    "status": "cancelled",
                }
                prior = assignments[command.owner]
                assignment = {
                    **prior,
                    "status": "cancelled",
                    "placement": {
                        **prior["placement"],
                        "reservation": cancelled_reservation,
                    },
                }
                reservations[command.owner] = cancelled_reservation
            assignments[command.owner] = assignment
            results.append(assignment)
        reservation_state = expected_snapshot(
            {"worker-a": {"slots": 1}, "worker-b": {"slots": 1}},
            tuple(reservations.values()),
        )
        state = {
            "assignments": tuple(
                sorted(
                    assignments.values(),
                    key=lambda item: (
                        item["frames"][0],
                        item["placement"]["reservation"]["id"],
                    ),
                )
            ),
            "reservations": reservation_state,
        }
        result_tuple = tuple(results)
        return ApplicationOutcome(
            result_tuple, state, _counters(operations, result_tuple, state)
        )

    return run_application_case(
        scenario_id="render-farm-frames",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    return run_benchmark(operations=operations, seed=DEFAULT_SEED)


@pytest.mark.benchmark
def test_render_farm_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
