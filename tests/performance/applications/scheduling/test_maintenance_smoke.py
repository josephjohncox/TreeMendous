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
from treemendous.applications.scheduling.maintenance import (
    MaintenanceBooking,
    MaintenanceScheduler,
    MaintenanceService,
)
from treemendous.domain import Span


def _booking_evidence(booking: MaintenanceBooking):
    return {
        "task_id": booking.task_id,
        "service": booking.service,
        "dependencies": booking.dependencies,
        "reservation": reservation_evidence(booking.reservation),
    }


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    window = Span(0, operations * 4 + 10)
    services = (
        MaintenanceService("api", (window,)),
        MaintenanceService("database", (window,)),
    )
    scheduler = MaintenanceScheduler(services)
    commands: list[SchedulingCommand] = []
    active: list[int] = []
    for step in make_plan(operations, seed):
        owner = f"task-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}))
            active.remove(step.ordinal)
            continue
        dependencies = (f"task-{active[-1]}",) if active else ()
        active.append(step.ordinal)
        start = step.ordinal * 4 + step.variant % 2
        service = ("api", "database")[step.variant % 2]
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "service": service,
                    "duration": 1,
                    "dependencies": dependencies,
                    "earliest_start": start,
                    "latest_end": start + 1,
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def invoke(engine: MaintenanceScheduler, command: SchedulingCommand):
        if command.action == "cancel":
            return engine.cancel(command.owner)
        return engine.schedule(command.owner, **command.arguments)

    def reserve_record(command: SchedulingCommand):
        arguments = command.arguments
        return expected_reservation(
            owner=command.owner,
            start=arguments["earliest_start"],
            end=arguments["latest_end"],
            requirements=((arguments["service"], {"slots": 1}),),
            request_id=arguments["request_id"],
        )

    def result_record(command: SchedulingCommand, record):
        arguments = next(
            item.arguments
            for item in prepared
            if item.action == "reserve" and item.owner == command.owner
        )
        return {
            "task_id": command.owner,
            "service": arguments["service"],
            "dependencies": tuple(sorted(arguments["dependencies"])),
            "reservation": record,
        }

    def oracle():
        return reservation_oracle_outcome(
            operations=operations,
            commands=prepared,
            resources={"api": {"slots": 1}, "database": {"slots": 1}},
            reserve_record=reserve_record,
            result_record=result_record,
        )

    return run_reservation_case(
        scenario_id="scheduling-maintenance",
        operations=operations,
        scheduler=scheduler,
        commands=prepared,
        invoke=invoke,
        result_evidence=_booking_evidence,
        oracle=oracle,
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    return run_benchmark(operations=operations, seed=DEFAULT_SEED)


@pytest.mark.benchmark
def test_maintenance_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
