import pytest

from tests.performance.applications.harness import ApplicationSample
from tests.performance.applications.scheduling._shared import (
    DEFAULT_OPERATIONS,
    DEFAULT_SEED,
    SchedulingCommand,
    expected_reservation,
    make_plan,
    placement_evidence,
    reservation_oracle_outcome,
    run_reservation_case,
)
from treemendous.applications.scheduling.warehouse_docks import (
    Dock,
    WarehouseDockScheduler,
)


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    docks = (
        Dock("dry-dock", frozenset({"dry"})),
        Dock("cold-dock", frozenset({"cold"})),
    )
    scheduler = WarehouseDockScheduler(docks)
    commands: list[SchedulingCommand] = []
    for step in make_plan(operations, seed):
        owner = f"carrier-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}, f"{owner}:1"))
            continue
        cargo_type = ("dry", "cold")[step.variant % 2]
        start = step.ordinal * 4 + 2 + step.variant % 2
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "duration": 1,
                    "cargo_type": cargo_type,
                    "earliest_start": start,
                    "latest_end": start + 1,
                    "handling_before": 1,
                    "handling_after": 1,
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def invoke(engine: WarehouseDockScheduler, command: SchedulingCommand):
        if command.action == "cancel":
            return engine.cancel(command.owner, command.reservation_id)  # type: ignore[arg-type]
        return engine.book(command.owner, **command.arguments)

    def reserve_record(command: SchedulingCommand):
        arguments = command.arguments
        resource = f"{arguments['cargo_type']}-dock"
        return expected_reservation(
            owner=command.owner,
            start=arguments["earliest_start"],
            end=arguments["latest_end"],
            requirements=((resource, {"units": 1}),),
            request_id=arguments["request_id"],
            buffer_before=1,
            buffer_after=1,
        )

    def oracle():
        return reservation_oracle_outcome(
            operations=operations,
            commands=prepared,
            resources={"cold-dock": {"units": 1}, "dry-dock": {"units": 1}},
            reserve_record=reserve_record,
            result_record=lambda _command, record: {
                "resource": record["requirements"][0][0],
                "reservation": record,
            },
        )

    return run_reservation_case(
        scenario_id="warehouse-dock-appointments",
        operations=operations,
        scheduler=scheduler,
        commands=prepared,
        invoke=invoke,
        result_evidence=placement_evidence,
        oracle=oracle,
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    return run_benchmark(operations=operations, seed=DEFAULT_SEED)


@pytest.mark.benchmark
def test_warehouse_dock_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
