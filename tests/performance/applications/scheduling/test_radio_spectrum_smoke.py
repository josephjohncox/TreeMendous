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
    make_plan,
)
from treemendous.applications.scheduling.radio_spectrum import (
    RadioSpectrumScheduler,
    SpectrumReservation,
    SpectrumSnapshot,
)

CHANNEL_COUNT = 32


def _reservation_evidence(reservation: SpectrumReservation):
    return {
        "id": reservation.id,
        "owner": reservation.owner,
        "channel_start": reservation.channel_start,
        "channel_end": reservation.channel_end,
        "start": reservation.start,
        "end": reservation.end,
        "guard_channels": reservation.guard_channels,
        "request_id": reservation.request_id,
        "status": reservation.status.value,
    }


def _snapshot_evidence(snapshot: SpectrumSnapshot):
    return {
        "reservations": tuple(
            _reservation_evidence(item) for item in snapshot.reservations
        ),
        "geometry": {
            "dimensions": snapshot.geometry.dimensions,
            "version": snapshot.geometry.version,
            "entries": tuple(
                {
                    "sequence": entry.handle.sequence,
                    "lower": entry.box.lower,
                    "upper": entry.box.upper,
                    "data": entry.data,
                }
                for entry in snapshot.geometry.entries
            ),
        },
    }


def _counters(operations: int, results, state):
    reservations = state["reservations"]
    active = tuple(item for item in reservations if item["status"] == "active")
    return {
        "operations": operations,
        "results": len(results),
        "reservations": len(reservations),
        "active": len(active),
        "cancelled": len(reservations) - len(active),
        "reserved_channel_slots": sum(
            (item["channel_end"] - item["channel_start"])
            * (item["end"] - item["start"])
            for item in reservations
        ),
        "active_guarded_channel_slots": sum(
            (
                min(CHANNEL_COUNT, item["channel_end"] + item["guard_channels"])
                - max(0, item["channel_start"] - item["guard_channels"])
            )
            * (item["end"] - item["start"])
            for item in active
        ),
        "geometry_entries": len(state["geometry"]["entries"]),
        "geometry_version": state["geometry"]["version"],
        "next_owner_sequences": tuple(
            sorted((item["owner"], 2) for item in reservations)
        ),
    }


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    scheduler = RadioSpectrumScheduler(CHANNEL_COUNT)
    commands: list[SchedulingCommand] = []
    for step in make_plan(operations, seed):
        owner = f"transmitter-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}, f"{owner}:1"))
            continue
        channel_start = (2, 10, 18)[step.variant % 3]
        start = step.ordinal * 3 + step.variant % 2
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "channel_start": channel_start,
                    "channel_width": 3,
                    "start": start,
                    "end": start + 1,
                    "guard_channels": (step.variant // 3) % 2,
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def execute() -> list[SpectrumReservation]:
        results = []
        for command in prepared:
            if command.action == "cancel":
                results.append(scheduler.cancel(command.owner, command.reservation_id))  # type: ignore[arg-type]
            else:
                results.append(scheduler.reserve(command.owner, **command.arguments))
        return results

    def observe(raw: list[SpectrumReservation]) -> ApplicationOutcome:
        results = tuple(_reservation_evidence(item) for item in raw)
        state = _snapshot_evidence(scheduler.snapshot())
        return ApplicationOutcome(results, state, _counters(operations, results, state))

    def oracle() -> ApplicationOutcome:
        records = {}
        entries = {}
        results = []
        version = 0
        sequence = 0
        for command in prepared:
            if command.action == "reserve":
                sequence += 1
                arguments = command.arguments
                channel_end = arguments["channel_start"] + arguments["channel_width"]
                record = {
                    "id": f"{command.owner}:1",
                    "owner": command.owner,
                    "channel_start": arguments["channel_start"],
                    "channel_end": channel_end,
                    "start": arguments["start"],
                    "end": arguments["end"],
                    "guard_channels": arguments["guard_channels"],
                    "request_id": arguments["request_id"],
                    "status": "active",
                }
                records[command.owner] = record
                guard = arguments["guard_channels"]
                entries[command.owner] = {
                    "sequence": sequence,
                    "lower": (
                        max(0, arguments["channel_start"] - guard),
                        arguments["start"],
                    ),
                    "upper": (
                        min(CHANNEL_COUNT, channel_end + guard),
                        arguments["end"],
                    ),
                    "data": record["id"],
                }
            else:
                record = {**records[command.owner], "status": "cancelled"}
                records[command.owner] = record
                del entries[command.owner]
            version += 1
            results.append(record)
        state = {
            "reservations": tuple(
                sorted(
                    records.values(),
                    key=lambda item: (item["start"], item["channel_start"], item["id"]),
                )
            ),
            "geometry": {
                "dimensions": 2,
                "version": version,
                "entries": tuple(entries.values()),
            },
        }
        result_tuple = tuple(results)
        return ApplicationOutcome(
            result_tuple, state, _counters(operations, result_tuple, state)
        )

    return run_application_case(
        scenario_id="radio-spectrum-timeslots",
        operations=operations,
        execute=execute,
        observe=observe,
        oracle=oracle,
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    return run_benchmark(operations=operations, seed=DEFAULT_SEED)


@pytest.mark.benchmark
def test_radio_spectrum_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
