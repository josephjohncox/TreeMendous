import pytest

from tests.performance.applications.harness import ApplicationOutcome, ApplicationSample
from tests.performance.applications.scheduling._shared import (
    DEFAULT_OPERATIONS,
    DEFAULT_SEED,
    SchedulingCommand,
    expected_reservation,
    expected_snapshot,
    make_plan,
    placement_evidence,
    run_reservation_case,
    scheduling_counters,
)
from treemendous.applications.scheduling.airline_gates import AirlineGateScheduler, Gate


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    gates = (
        Gate("gate-a", frozenset({"A320"})),
        Gate("gate-b", frozenset({"B737"})),
    )
    scheduler = AirlineGateScheduler(gates)
    commands: list[SchedulingCommand] = []
    for step in make_plan(operations, seed):
        owner = f"flight-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}, f"{owner}:1"))
            continue
        arrival = step.ordinal * 4 + 2 + step.variant % 2
        aircraft_type = ("A320", "B737")[step.variant % 2]
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "arrival": arrival,
                    "departure": arrival + 1,
                    "aircraft_type": aircraft_type,
                    "turnaround_before": 1,
                    "turnaround_after": 1,
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def invoke(engine: AirlineGateScheduler, command: SchedulingCommand):
        if command.action == "cancel":
            return engine.cancel(command.owner, command.reservation_id)  # type: ignore[arg-type]
        return engine.assign(command.owner, **command.arguments)

    def oracle() -> ApplicationOutcome:
        cancelled = {item.owner for item in prepared if item.action == "cancel"}
        reservations = []
        results = []
        for command in prepared:
            if command.action == "reserve":
                arguments = command.arguments
                resource = (
                    "gate-a" if arguments["aircraft_type"] == "A320" else "gate-b"
                )
                record = expected_reservation(
                    owner=command.owner,
                    start=arguments["arrival"],
                    end=arguments["departure"],
                    requirements=((resource, {"units": 1}),),
                    request_id=arguments["request_id"],
                    buffer_before=1,
                    buffer_after=1,
                )
                reservations.append(
                    {
                        **record,
                        "status": (
                            "cancelled" if command.owner in cancelled else "active"
                        ),
                    }
                )
                results.append({"resource": resource, "reservation": record})
            else:
                original = next(
                    item for item in reservations if item["owner"] == command.owner
                )
                results.append(
                    {
                        "resource": original["requirements"][0][0],
                        "reservation": {**original, "status": "cancelled"},
                    }
                )
        state = expected_snapshot(
            {"gate-a": {"units": 1}, "gate-b": {"units": 1}}, reservations
        )
        result_tuple = tuple(results)
        return ApplicationOutcome(
            result_tuple,
            state,
            scheduling_counters(
                operations=operations, results=result_tuple, state=state
            ),
        )

    return run_reservation_case(
        scenario_id="airline-gate-assignment",
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
def test_airline_gate_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
