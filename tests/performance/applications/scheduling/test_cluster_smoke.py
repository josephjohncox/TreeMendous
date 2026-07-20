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
from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications.scheduling.cluster import ClusterNode, ClusterScheduler


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    nodes = (
        ClusterNode("cpu-node", CapacityVector(cpu=4, memory=16), frozenset({"cpu"})),
        ClusterNode("gpu-node", CapacityVector(cpu=4, memory=16), frozenset({"gpu"})),
    )
    scheduler = ClusterScheduler(nodes)
    commands: list[SchedulingCommand] = []
    for step in make_plan(operations, seed):
        owner = f"job-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}, f"{owner}:1"))
            continue
        start = step.ordinal * 3 + step.variant % 2
        label = ("cpu", "gpu")[step.variant % 2]
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "duration": 1,
                    "demand": CapacityVector(cpu=1, memory=2),
                    "required_labels": frozenset({label}),
                    "earliest_start": start,
                    "latest_end": start + 1,
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def invoke(engine: ClusterScheduler, command: SchedulingCommand):
        if command.action == "cancel":
            return engine.cancel(command.owner, command.reservation_id)  # type: ignore[arg-type]
        return engine.schedule(command.owner, **command.arguments)

    def reserve_record(command: SchedulingCommand):
        arguments = command.arguments
        label = next(iter(arguments["required_labels"]))
        return expected_reservation(
            owner=command.owner,
            start=arguments["earliest_start"],
            end=arguments["latest_end"],
            requirements=((f"{label}-node", {"cpu": 1, "memory": 2}),),
            request_id=arguments["request_id"],
        )

    def oracle():
        return reservation_oracle_outcome(
            operations=operations,
            commands=prepared,
            resources={
                "cpu-node": {"cpu": 4, "memory": 16},
                "gpu-node": {"cpu": 4, "memory": 16},
            },
            reserve_record=reserve_record,
            result_record=lambda _command, record: {
                "resource": record["requirements"][0][0],
                "reservation": record,
            },
        )

    return run_reservation_case(
        scenario_id="distributed-cluster-scheduling",
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
def test_cluster_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
