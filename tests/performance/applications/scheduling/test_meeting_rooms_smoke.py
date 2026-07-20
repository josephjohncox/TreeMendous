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
from treemendous.applications.scheduling._common import Placement
from treemendous.applications.scheduling.meeting_rooms import (
    MeetingBooking,
    MeetingRoom,
    MeetingRoomScheduler,
)


def _meeting_evidence(result: MeetingBooking | Placement):
    if isinstance(result, MeetingBooking):
        return {
            "placement": placement_evidence(result.placement),
            "attendees": result.attendees,
            "utc_span": (result.utc_span.start, result.utc_span.end),
        }
    return placement_evidence(result)


def run_benchmark(
    operations: int = DEFAULT_OPERATIONS, seed: int = DEFAULT_SEED
) -> ApplicationSample:
    rooms = (
        MeetingRoom("board-room", 8, frozenset({"whiteboard"})),
        MeetingRoom("video-room", 4, frozenset({"video"})),
    )
    scheduler = MeetingRoomScheduler(rooms)
    commands: list[SchedulingCommand] = []
    for step in make_plan(operations, seed):
        owner = f"meeting-{step.ordinal}"
        if step.action == "cancel":
            commands.append(SchedulingCommand("cancel", owner, {}, f"{owner}:1"))
            continue
        variant = step.variant % 2
        feature = ("video", "whiteboard")[variant]
        attendees = (4, 6)[variant]
        offset = (-2, 3)[(step.variant // 2) % 2]
        utc_start = step.ordinal * 3 + variant
        commands.append(
            SchedulingCommand(
                "reserve",
                owner,
                {
                    "local_start": utc_start + offset,
                    "local_end": utc_start + offset + 1,
                    "timezone_offset_slots": offset,
                    "attendees": attendees,
                    "features": frozenset({feature}),
                    "request_id": f"request-{step.ordinal}",
                },
            )
        )
    prepared = tuple(commands)

    def invoke(engine: MeetingRoomScheduler, command: SchedulingCommand):
        if command.action == "cancel":
            return engine.cancel(command.owner, command.reservation_id)  # type: ignore[arg-type]
        return engine.book(command.owner, **command.arguments)

    def reserve_record(command: SchedulingCommand):
        arguments = command.arguments
        feature = next(iter(arguments["features"]))
        resource = "video-room" if feature == "video" else "board-room"
        start = arguments["local_start"] - arguments["timezone_offset_slots"]
        end = arguments["local_end"] - arguments["timezone_offset_slots"]
        return expected_reservation(
            owner=command.owner,
            start=start,
            end=end,
            requirements=((resource, {"units": 1}),),
            request_id=arguments["request_id"],
        )

    def result_record(command: SchedulingCommand, record):
        placement = {
            "resource": record["requirements"][0][0],
            "reservation": record,
        }
        if command.action == "cancel":
            return placement
        arguments = command.arguments
        return {
            "placement": placement,
            "attendees": arguments["attendees"],
            "utc_span": (record["start"], record["end"]),
        }

    def oracle():
        return reservation_oracle_outcome(
            operations=operations,
            commands=prepared,
            resources={"board-room": {"units": 1}, "video-room": {"units": 1}},
            reserve_record=reserve_record,
            result_record=result_record,
        )

    return run_reservation_case(
        scenario_id="meeting-room-booking",
        operations=operations,
        scheduler=scheduler,
        commands=prepared,
        invoke=invoke,
        result_evidence=_meeting_evidence,
        oracle=oracle,
    )


def run_smoke(operations: int = DEFAULT_OPERATIONS) -> ApplicationSample:
    return run_benchmark(operations=operations, seed=DEFAULT_SEED)


@pytest.mark.benchmark
def test_meeting_room_smoke_matches_oracle() -> None:
    sample = run_benchmark(operations=16, seed=DEFAULT_SEED)
    assert sample.validated
    assert sample.result_checksum and sample.state_checksum
    assert sample.counters_checksum and sample.evidence_checksum
