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
from treemendous.applications.scheduling.fleet_charging import (
    Charger,
    ChargingSession,
    FleetChargingScheduler,
)


def _session_evidence(session: ChargingSession):
    return {
        "charger": session.charger,
        "energy": session.energy,
        "power_per_slot": session.power_per_slot,
        "duration": session.duration,
        "reservation": reservation_evidence(session.reservation),
    }


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    chargers = (
        Charger("ccs-charger", 10, frozenset({"ccs"})),
        Charger("nacs-charger", 20, frozenset({"nacs"})),
    )
    scheduler = FleetChargingScheduler(chargers, max_session_slots=3)
    commands: list[SchedulingCommand] = []
    for step in make_plan(operations, seed):
        owner = f"vehicle-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}, f"{owner}:1"))
            continue
        connector = ("ccs", "nacs")[step.variant % 2]
        power = 10 if connector == "ccs" else 20
        duration = 1 + (step.variant // 2) % 2
        arrival = step.ordinal * 5 + step.variant % 2
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "energy": power * duration,
                    "connector": connector,
                    "arrival": arrival,
                    "departure": arrival + duration,
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def invoke(engine: FleetChargingScheduler, command: SchedulingCommand):
        if command.action == "cancel":
            return engine.cancel(command.owner, command.reservation_id)  # type: ignore[arg-type]
        return engine.schedule(command.owner, **command.arguments)

    def reserve_record(command: SchedulingCommand):
        arguments = command.arguments
        connector = arguments["connector"]
        power = 10 if connector == "ccs" else 20
        duration = (arguments["energy"] + power - 1) // power
        return expected_reservation(
            owner=command.owner,
            start=arguments["arrival"],
            end=arguments["arrival"] + duration,
            requirements=((f"{connector}-charger", {"units": 1}),),
            request_id=arguments["request_id"],
        )

    def result_record(command: SchedulingCommand, record):
        arguments = next(
            item.arguments
            for item in prepared
            if item.action == "reserve" and item.owner == command.owner
        )
        power = 10 if arguments["connector"] == "ccs" else 20
        return {
            "charger": f"{arguments['connector']}-charger",
            "energy": arguments["energy"],
            "power_per_slot": power,
            "duration": record["end"] - record["start"],
            "reservation": record,
        }

    def oracle():
        return reservation_oracle_outcome(
            operations=operations,
            commands=prepared,
            resources={
                "ccs-charger": {"units": 1},
                "nacs-charger": {"units": 1},
            },
            reserve_record=reserve_record,
            result_record=result_record,
        )

    return run_reservation_case(
        scenario_id="scheduling-fleet-charging",
        operations=operations,
        scheduler=scheduler,
        commands=prepared,
        invoke=invoke,
        result_evidence=_session_evidence,
        oracle=oracle,
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    return run_benchmark(operations=operations, seed=DEFAULT_SEED)


@pytest.mark.benchmark
def test_fleet_charging_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
