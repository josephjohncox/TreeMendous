import pytest

from tests.performance.applications.harness import ApplicationSample
from tests.performance.applications.scheduling._shared import (
    DEFAULT_OPERATIONS,
    DEFAULT_SEED,
    SchedulingCommand,
    expected_reservation,
    make_plan,
    reservation_evidence,
    reservation_oracle_outcome,
    run_reservation_case,
)
from treemendous.applications.scheduling.operating_rooms import (
    ClinicalResource,
    OperatingRoom,
    OperatingRoomScheduler,
    ProcedureBooking,
)


def _procedure_evidence(booking: ProcedureBooking):
    return {
        "room": booking.room,
        "staff": booking.staff,
        "equipment": booking.equipment,
        "reservation": reservation_evidence(booking.reservation),
    }


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    rooms = (
        OperatingRoom("cardiac-or", frozenset({"cardiac"})),
        OperatingRoom("general-or", frozenset({"general"})),
    )
    staff = (ClinicalResource("surgeon-a"), ClinicalResource("surgeon-b"))
    equipment = (ClinicalResource("monitor-a"), ClinicalResource("monitor-b"))
    scheduler = OperatingRoomScheduler(rooms, staff, equipment)
    commands: list[SchedulingCommand] = []
    for step in make_plan(operations, seed):
        owner = f"procedure-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}, f"{owner}:1"))
            continue
        variant = step.variant % 2
        capability = ("general", "cardiac")[variant]
        suffix = ("a", "b")[variant]
        start = step.ordinal * 3 + variant
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "duration": 1,
                    "room_capabilities": frozenset({capability}),
                    "staff": (f"surgeon-{suffix}",),
                    "equipment": (f"monitor-{suffix}",),
                    "earliest_start": start,
                    "latest_end": start + 1,
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def invoke(engine: OperatingRoomScheduler, command: SchedulingCommand):
        if command.action == "cancel":
            return engine.cancel(command.owner, command.reservation_id)  # type: ignore[arg-type]
        return engine.schedule(command.owner, **command.arguments)

    def reserve_record(command: SchedulingCommand):
        arguments = command.arguments
        capability = next(iter(arguments["room_capabilities"]))
        room = f"{capability}-or"
        return expected_reservation(
            owner=command.owner,
            start=arguments["earliest_start"],
            end=arguments["latest_end"],
            requirements=(
                (f"room:{room}", {"units": 1}),
                (f"staff:{arguments['staff'][0]}", {"units": 1}),
                (f"equipment:{arguments['equipment'][0]}", {"units": 1}),
            ),
            request_id=arguments["request_id"],
        )

    def result_record(_command: SchedulingCommand, record):
        resources = tuple(item[0] for item in record["requirements"])
        return {
            "room": next(
                item.removeprefix("room:")
                for item in resources
                if item.startswith("room:")
            ),
            "staff": tuple(
                item.removeprefix("staff:")
                for item in resources
                if item.startswith("staff:")
            ),
            "equipment": tuple(
                item.removeprefix("equipment:")
                for item in resources
                if item.startswith("equipment:")
            ),
            "reservation": record,
        }

    def oracle():
        resources = {
            "room:cardiac-or": {"units": 1},
            "room:general-or": {"units": 1},
            "staff:surgeon-a": {"units": 1},
            "staff:surgeon-b": {"units": 1},
            "equipment:monitor-a": {"units": 1},
            "equipment:monitor-b": {"units": 1},
        }
        return reservation_oracle_outcome(
            operations=operations,
            commands=prepared,
            resources=resources,
            reserve_record=reserve_record,
            result_record=result_record,
        )

    return run_reservation_case(
        scenario_id="scheduling-operating-rooms",
        operations=operations,
        scheduler=scheduler,
        commands=prepared,
        invoke=invoke,
        result_evidence=_procedure_evidence,
        oracle=oracle,
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    return run_benchmark(operations=operations, seed=DEFAULT_SEED)


@pytest.mark.benchmark
def test_operating_room_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
