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
from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications.scheduling.gpu_streams import (
    GPUDevice,
    GPUPlacement,
    GPUStream,
    GPUStreamScheduler,
)


def _gpu_evidence(placement: GPUPlacement):
    return {
        "device": placement.device,
        "stream": placement.stream,
        "reservation": reservation_evidence(placement.reservation),
    }


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    devices = (
        GPUDevice(
            "compute-gpu",
            CapacityVector(memory=8, slots=2),
            frozenset({"compute"}),
            (GPUStream("compute-stream", frozenset({"compute"})),),
        ),
        GPUDevice(
            "graphics-gpu",
            CapacityVector(memory=8, slots=2),
            frozenset({"graphics"}),
            (GPUStream("graphics-stream", frozenset({"graphics"})),),
        ),
    )
    scheduler = GPUStreamScheduler(devices)
    commands: list[SchedulingCommand] = []
    for step in make_plan(operations, seed):
        owner = f"kernel-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}, f"{owner}:1"))
            continue
        compatibility = ("compute", "graphics")[step.variant % 2]
        start = step.ordinal * 3 + step.variant % 2
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "duration": 1,
                    "demand": CapacityVector(memory=1, slots=1),
                    "compatibility": compatibility,
                    "dependency_ready_times": {"input": start},
                    "earliest_start": start - 1,
                    "latest_end": start + 1,
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def invoke(engine: GPUStreamScheduler, command: SchedulingCommand):
        if command.action == "cancel":
            return engine.cancel(command.owner, command.reservation_id)  # type: ignore[arg-type]
        return engine.schedule(command.owner, **command.arguments)

    def reserve_record(command: SchedulingCommand):
        arguments = command.arguments
        kind = arguments["compatibility"]
        device = f"{kind}-gpu"
        stream = f"{kind}-stream"
        start = arguments["dependency_ready_times"]["input"]
        return expected_reservation(
            owner=command.owner,
            start=start,
            end=start + 1,
            requirements=(
                (f"device:{device}", {"memory": 1, "slots": 1}),
                (f"stream:{device}:{stream}", {"units": 1}),
            ),
            request_id=arguments["request_id"],
        )

    def result_record(_command: SchedulingCommand, record):
        device = record["requirements"][0][0].removeprefix("device:")
        stream = record["requirements"][1][0].split(":", maxsplit=2)[2]
        return {"device": device, "stream": stream, "reservation": record}

    def oracle():
        return reservation_oracle_outcome(
            operations=operations,
            commands=prepared,
            resources={
                "device:compute-gpu": {"memory": 8, "slots": 2},
                "stream:compute-gpu:compute-stream": {"units": 1},
                "device:graphics-gpu": {"memory": 8, "slots": 2},
                "stream:graphics-gpu:graphics-stream": {"units": 1},
            },
            reserve_record=reserve_record,
            result_record=result_record,
        )

    return run_reservation_case(
        scenario_id="scheduling-gpu-streams",
        operations=operations,
        scheduler=scheduler,
        commands=prepared,
        invoke=invoke,
        result_evidence=_gpu_evidence,
        oracle=oracle,
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    return run_benchmark(operations=operations, seed=DEFAULT_SEED)


@pytest.mark.benchmark
def test_gpu_stream_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
