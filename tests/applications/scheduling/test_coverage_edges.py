"""Public edge and adversarial coverage for scheduling applications."""

from __future__ import annotations

import pytest

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import ReservationStatus
from treemendous.applications.scheduling._common import SchedulingUnavailableError
from treemendous.applications.scheduling.cluster import ClusterNode, ClusterScheduler
from treemendous.applications.scheduling.fleet_charging import (
    Charger,
    FleetChargingScheduler,
    create_fleet_charging_scheduler,
)
from treemendous.applications.scheduling.gpu_streams import (
    GPUDevice,
    GPUStream,
    GPUStreamScheduler,
    create_gpu_stream_scheduler,
)
from treemendous.applications.scheduling.lab_instruments import (
    LabInstrument,
    LabInstrumentScheduler,
    create_lab_instrument_scheduler,
)
from treemendous.applications.scheduling.maintenance import (
    MaintenanceScheduler,
    MaintenanceService,
    create_maintenance_scheduler,
)
from treemendous.applications.scheduling.operating_rooms import (
    ClinicalResource,
    OperatingRoom,
    OperatingRoomScheduler,
    create_operating_room_scheduler,
)
from treemendous.applications.scheduling.radio_spectrum import (
    RadioSpectrumScheduler,
    SpectrumConflictError,
    SpectrumStatus,
    create_radio_spectrum_scheduler,
)
from treemendous.applications.scheduling.render_farm import (
    ChunkStatus,
    RenderFarmScheduler,
    RenderWorker,
    create_render_farm_scheduler,
)
from treemendous.domain import Span


def test_cluster_public_validation_and_no_eligible_resource_edges() -> None:
    with pytest.raises(TypeError, match="CapacityVector"):
        ClusterNode("bad", {"slots": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one resource"):
        ClusterScheduler(())

    node = ClusterNode("node", CapacityVector(cpu=2), frozenset({"linux"}))
    with pytest.raises(ValueError, match="unique"):
        ClusterScheduler((node, node))
    scheduler = ClusterScheduler((node,))

    with pytest.raises(TypeError, match="demand must be"):
        scheduler.schedule("job", 1, {"cpu": 1}, latest_end=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bounded window"):
        scheduler.schedule("job", 2, CapacityVector(cpu=1), latest_end=1)
    with pytest.raises(SchedulingUnavailableError) as wrong_labels:
        scheduler.schedule(
            "job",
            1,
            CapacityVector(cpu=1),
            required_labels=frozenset({"gpu"}),
            latest_end=1,
        )
    assert wrong_labels.value.considered == ("node",)
    with pytest.raises(SchedulingUnavailableError):
        scheduler.schedule("job", 1, CapacityVector(memory=1), latest_end=1)
    with pytest.raises(SchedulingUnavailableError):
        scheduler.schedule("job", 1, CapacityVector(cpu=3), latest_end=1)


def test_cluster_idempotency_conflict_diagnostics_and_cancel_replay() -> None:
    scheduler = ClusterScheduler(
        (ClusterNode("node", CapacityVector(slots=1), frozenset({"linux"})),)
    )
    first = scheduler.schedule(
        "job",
        2,
        CapacityVector(slots=1),
        required_labels=frozenset({"linux"}),
        latest_end=2,
        request_id="once",
    )
    assert first.start == 0
    assert first.end == 2
    assert first.resource == "node"

    with pytest.raises(ValueError, match="idempotency key"):
        scheduler.schedule(
            "job",
            1,
            CapacityVector(slots=1),
            required_labels=frozenset({"linux"}),
            latest_end=2,
            request_id="once",
        )
    with pytest.raises(SchedulingUnavailableError) as unavailable:
        scheduler.schedule(
            "blocked", 2, CapacityVector(slots=1), earliest_start=0, latest_end=2
        )
    assert unavailable.value.conflicts
    assert unavailable.value.considered == ("node",)
    assert scheduler.snapshot().reservations == (first.reservation,)

    cancelled = scheduler.cancel("job", first.id)
    assert cancelled.reservation.status is ReservationStatus.CANCELLED
    assert (
        scheduler.schedule(
            "job",
            2,
            CapacityVector(slots=1),
            required_labels=frozenset({"linux"}),
            latest_end=2,
            request_id="once",
        )
        is cancelled
    )


def test_lab_constructor_validation_and_factory() -> None:
    with pytest.raises(ValueError, match="nonempty tuple"):
        LabInstrument("scope", frozenset(), ())
    with pytest.raises(TypeError, match="Span values"):
        LabInstrument("scope", frozenset(), ("always",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 0"):
        LabInstrument("scope", frozenset(), (Span(0, 1),), cleanup_slots=-1)
    with pytest.raises(ValueError, match="at least one instrument"):
        LabInstrumentScheduler(())

    instrument = LabInstrument("scope", frozenset({"image"}), (Span(0, 5),))
    with pytest.raises(ValueError, match="unique"):
        LabInstrumentScheduler((instrument, instrument))
    default = create_lab_instrument_scheduler()
    assert default.snapshot().resources[0].resource == "scope-a"


def test_lab_calibration_conflicts_idempotency_cancel_and_snapshot() -> None:
    scheduler = LabInstrumentScheduler(
        (
            LabInstrument("a", frozenset({"image"}), (Span(4, 10),), 1),
            LabInstrument("b", frozenset({"image"}), (Span(0, 3),), 0),
        )
    )
    first = scheduler.book(
        "experiment",
        2,
        capabilities=frozenset({"image"}),
        earliest_start=0,
        latest_end=10,
        request_id="book",
    )
    assert (first.resource, first.start, first.reservation.occupied_span.end) == (
        "b",
        0,
        2,
    )
    replay = scheduler.book(
        "experiment",
        2,
        capabilities=frozenset({"image"}),
        earliest_start=0,
        latest_end=10,
        request_id="book",
    )
    assert replay is first
    before = scheduler.snapshot()
    with pytest.raises(ValueError, match="idempotency key"):
        scheduler.book(
            "experiment",
            1,
            capabilities=frozenset({"image"}),
            earliest_start=0,
            latest_end=10,
            request_id="book",
        )
    assert scheduler.snapshot() == before

    blocker = scheduler.book(
        "blocker",
        5,
        capabilities=frozenset({"image"}),
        earliest_start=4,
        latest_end=10,
    )
    with pytest.raises(SchedulingUnavailableError) as unavailable:
        scheduler.book(
            "later",
            2,
            capabilities=frozenset({"image"}),
            earliest_start=4,
            latest_end=10,
        )
    assert unavailable.value.conflicts
    assert unavailable.value.considered == ("a", "b")

    cancelled = scheduler.cancel("blocker", blocker.id)
    assert not cancelled.reservation.active
    assert scheduler.cancel("blocker", blocker.id) == cancelled
    assert scheduler.snapshot().reservations[-1] == cancelled.reservation


def test_operating_room_constructor_and_request_validation() -> None:
    with pytest.raises(ValueError, match="at least one operating room"):
        OperatingRoomScheduler((), (), ())
    room = OperatingRoom("or", frozenset({"robotic"}))
    staff = ClinicalResource("surgeon")
    equipment = ClinicalResource("robot")
    with pytest.raises(ValueError, match="operating room names"):
        OperatingRoomScheduler((room, room), (), ())
    with pytest.raises(ValueError, match="clinical resource names"):
        OperatingRoomScheduler((room,), (staff, staff), ())
    with pytest.raises(ValueError, match="clinical resource names"):
        OperatingRoomScheduler((room,), (), (equipment, equipment))

    scheduler = OperatingRoomScheduler((room,), (staff,), (equipment,))
    common = dict(earliest_start=0, latest_end=2)
    with pytest.raises(ValueError, match="must be unique"):
        scheduler.schedule("p", 1, staff=("surgeon", "surgeon"), equipment=(), **common)
    with pytest.raises(KeyError, match="unknown staff"):
        scheduler.schedule("p", 1, staff=("missing",), equipment=(), **common)
    with pytest.raises(KeyError, match="unknown equipment"):
        scheduler.schedule("p", 1, staff=(), equipment=("missing",), **common)
    with pytest.raises(SchedulingUnavailableError, match="compatible"):
        scheduler.schedule(
            "p",
            1,
            room_capabilities=frozenset({"cardiac"}),
            staff=(),
            equipment=(),
            **common,
        )


def test_operating_room_idempotency_atomic_conflicts_and_cancel() -> None:
    scheduler = OperatingRoomScheduler(
        (
            OperatingRoom("a", frozenset({"general"})),
            OperatingRoom("b", frozenset({"general"})),
        ),
        (ClinicalResource("nurse"),),
        (ClinicalResource("monitor"),),
    )
    first = scheduler.schedule(
        "first",
        2,
        staff=("nurse",),
        equipment=("monitor",),
        earliest_start=0,
        latest_end=2,
        request_id="once",
    )
    assert first.id == first.reservation.id
    assert (first.room, first.staff, first.equipment) == (
        "a",
        ("nurse",),
        ("monitor",),
    )
    assert (
        scheduler.schedule(
            "first",
            2,
            staff=("nurse",),
            equipment=("monitor",),
            earliest_start=0,
            latest_end=2,
            request_id="once",
        )
        is first
    )
    before = scheduler.snapshot()
    with pytest.raises(ValueError, match="idempotency key"):
        scheduler.schedule(
            "first",
            1,
            staff=("nurse",),
            equipment=("monitor",),
            earliest_start=0,
            latest_end=2,
            request_id="once",
        )
    with pytest.raises(SchedulingUnavailableError) as unavailable:
        scheduler.schedule(
            "blocked",
            2,
            staff=("nurse",),
            equipment=("monitor",),
            earliest_start=0,
            latest_end=2,
        )
    assert unavailable.value.conflicts
    assert scheduler.snapshot() == before

    cancelled = scheduler.cancel("first", first.id)
    assert not cancelled.reservation.active
    assert cancelled.room == "a"
    assert scheduler.cancel("first", first.id) == cancelled
    replay = scheduler.schedule(
        "first",
        2,
        staff=("nurse",),
        equipment=("monitor",),
        earliest_start=0,
        latest_end=2,
        request_id="once",
    )
    assert replay == cancelled
    default_reservations = create_operating_room_scheduler().snapshot().reservations
    assert not default_reservations


def test_fleet_constructor_validation_and_no_eligible_charger() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Charger("bad", 0, frozenset({"ccs"}))
    with pytest.raises(TypeError, match="frozenset"):
        Charger("bad", 1, {"ccs"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one connector"):
        Charger("bad", 1, frozenset())
    with pytest.raises(ValueError, match="at least one charger"):
        FleetChargingScheduler((), max_session_slots=1)
    charger = Charger("one", 2, frozenset({"ccs"}))
    with pytest.raises(ValueError, match="max_session_slots"):
        FleetChargingScheduler((charger,), max_session_slots=0)
    with pytest.raises(ValueError, match="unique"):
        FleetChargingScheduler((charger, charger), max_session_slots=2)

    scheduler = FleetChargingScheduler((charger,), max_session_slots=2)
    with pytest.raises(SchedulingUnavailableError) as incompatible:
        scheduler.schedule("car", 1, connector="nacs", arrival=0, departure=2)
    assert not incompatible.value.considered
    with pytest.raises(SchedulingUnavailableError):
        scheduler.schedule("car", 5, connector="ccs", arrival=0, departure=2)
    default_reservations = create_fleet_charging_scheduler().snapshot().reservations
    assert not default_reservations


def test_fleet_completion_order_conflicts_idempotency_and_cancel() -> None:
    scheduler = FleetChargingScheduler(
        (
            Charger("fast", 4, frozenset({"ccs"})),
            Charger("slow", 2, frozenset({"ccs"})),
        ),
        max_session_slots=4,
    )
    occupied = scheduler.schedule(
        "occupied", 4, connector="ccs", arrival=0, departure=1
    )
    assert occupied.charger == "fast"
    session = scheduler.schedule(
        "van", 4, connector="ccs", arrival=0, departure=4, request_id="once"
    )
    actual_session = (session.charger, session.duration, session.id)
    expected_session = ("slow", 2, "van:1")
    assert actual_session == expected_session
    assert (
        scheduler.schedule(
            "van", 4, connector="ccs", arrival=0, departure=4, request_id="once"
        )
        is session
    )
    before = scheduler.snapshot()
    with pytest.raises(ValueError, match="idempotency key"):
        scheduler.schedule(
            "van", 2, connector="ccs", arrival=0, departure=4, request_id="once"
        )
    assert scheduler.snapshot() == before

    cancelled = scheduler.cancel("van", session.id)
    assert cancelled.reservation.status is ReservationStatus.CANCELLED
    assert scheduler.cancel("van", session.id) == cancelled
    replay = scheduler.schedule(
        "van", 4, connector="ccs", arrival=0, departure=4, request_id="once"
    )
    assert replay == cancelled


def _gpu_device(
    name: str = "gpu", *, capacity: CapacityVector | None = None
) -> GPUDevice:
    return GPUDevice(
        name,
        capacity or CapacityVector(memory=8, slots=1),
        frozenset({"compute"}),
        (GPUStream("stream", frozenset({"compute"})),),
    )


def test_gpu_constructor_and_dependency_validation() -> None:
    with pytest.raises(TypeError, match="CapacityVector"):
        GPUDevice("gpu", {"slots": 1}, frozenset(), (GPUStream("s", frozenset()),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one stream"):
        GPUDevice("gpu", CapacityVector(slots=1), frozenset(), ())
    stream = GPUStream("s", frozenset({"compute"}))
    with pytest.raises(ValueError, match="stream names"):
        GPUDevice("gpu", CapacityVector(slots=1), frozenset(), (stream, stream))
    with pytest.raises(ValueError, match="at least one GPU"):
        GPUStreamScheduler(())
    device = _gpu_device()
    with pytest.raises(ValueError, match="device names"):
        GPUStreamScheduler((device, device))

    scheduler = GPUStreamScheduler((device,))
    with pytest.raises(TypeError, match="demand must be"):
        scheduler.schedule("k", 1, {"slots": 1}, compatibility="compute", latest_end=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dependency must not be empty"):
        scheduler.schedule(
            "k",
            1,
            CapacityVector(memory=1, slots=1),
            compatibility="compute",
            dependency_ready_times={"": 0},
            latest_end=1,
        )
    with pytest.raises(SchedulingUnavailableError, match="dependencies"):
        scheduler.schedule(
            "k",
            2,
            CapacityVector(memory=1, slots=1),
            compatibility="compute",
            dependency_ready_times={"copy": 2},
            latest_end=3,
        )


def test_gpu_pair_filtering_bounded_conflict_cancel_and_factory() -> None:
    devices = (
        GPUDevice(
            "a-wrong-label",
            CapacityVector(memory=8, slots=1),
            frozenset({"graphics"}),
            (GPUStream("s", frozenset({"compute"})),),
        ),
        GPUDevice(
            "b-wrong-dimensions",
            CapacityVector(slots=1),
            frozenset({"compute"}),
            (GPUStream("s", frozenset({"compute"})),),
        ),
        GPUDevice(
            "c-too-small",
            CapacityVector(memory=1, slots=1),
            frozenset({"compute"}),
            (GPUStream("s", frozenset({"compute"})),),
        ),
        GPUDevice(
            "d-stream-label",
            CapacityVector(memory=8, slots=1),
            frozenset({"compute"}),
            (GPUStream("s", frozenset({"graphics"})),),
        ),
    )
    scheduler = GPUStreamScheduler(devices)
    demand = CapacityVector(memory=2, slots=1)
    with pytest.raises(SchedulingUnavailableError, match="no compatible"):
        scheduler.schedule("none", 1, demand, compatibility="compute", latest_end=1)

    available = GPUStreamScheduler((_gpu_device(),))
    first = available.schedule(
        "kernel", 2, demand, compatibility="compute", latest_end=2, request_id="once"
    )
    with pytest.raises(SchedulingUnavailableError) as bounded:
        available.schedule("blocked", 2, demand, compatibility="compute", latest_end=2)
    assert bounded.value.conflicts
    with pytest.raises(KeyError):
        available.cancel("kernel", "missing:1")
    with pytest.raises(PermissionError):
        available.cancel("other", first.id)
    cancelled = available.cancel("kernel", first.id)
    assert available.cancel("kernel", first.id) == cancelled
    replay = available.schedule(
        "kernel", 2, demand, compatibility="compute", latest_end=2, request_id="once"
    )
    assert replay == cancelled
    default_reservations = create_gpu_stream_scheduler().snapshot().reservations
    assert not default_reservations


def test_maintenance_constructor_and_dependency_edges() -> None:
    with pytest.raises(ValueError, match="nonempty tuple"):
        MaintenanceService("api", ())
    with pytest.raises(TypeError, match="blackouts"):
        MaintenanceService("api", (Span(0, 2),), ("never",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="concurrency"):
        MaintenanceService("api", (Span(0, 2),), concurrency=0)
    with pytest.raises(ValueError, match="at least one maintenance"):
        MaintenanceScheduler(())
    service = MaintenanceService("api", (Span(0, 10),))
    with pytest.raises(ValueError, match="service names"):
        MaintenanceScheduler((service, service))

    scheduler = MaintenanceScheduler((service,))
    with pytest.raises(KeyError, match="unknown service"):
        scheduler.schedule("task", "missing", 1, latest_end=2)
    with pytest.raises(ValueError, match="cannot include"):
        scheduler.schedule("task", "api", 1, dependencies=("task",), latest_end=2)
    with pytest.raises(ValueError, match="must be unique"):
        scheduler.schedule("task", "api", 1, dependencies=("a", "a"), latest_end=2)
    with pytest.raises(KeyError, match="unknown dependency"):
        scheduler.schedule("task", "api", 1, dependencies=("missing",), latest_end=2)
    default_reservations = create_maintenance_scheduler().snapshot().reservations
    assert not default_reservations


def test_maintenance_windows_conflicts_idempotency_and_cancellation() -> None:
    scheduler = MaintenanceScheduler(
        (
            MaintenanceService(
                "api", (Span(2, 5), Span(7, 12)), (Span(3, 4),), concurrency=1
            ),
        )
    )
    first = scheduler.schedule(
        "first", "api", 2, earliest_start=0, latest_end=12, request_id="once"
    )
    actual_booking = (first.id, first.reservation.start)
    expected_booking = (first.reservation.id, 7)
    assert actual_booking == expected_booking
    assert (
        scheduler.schedule(
            "first", "api", 2, earliest_start=0, latest_end=12, request_id="once"
        )
        is first
    )
    before = scheduler.snapshot()
    with pytest.raises(ValueError, match="idempotency key"):
        scheduler.schedule(
            "first", "api", 1, earliest_start=0, latest_end=12, request_id="once"
        )
    with pytest.raises(ValueError, match="already scheduled"):
        scheduler.schedule("first", "api", 1, latest_end=12)
    with pytest.raises(SchedulingUnavailableError) as unavailable:
        scheduler.schedule("blocked", "api", 2, earliest_start=7, latest_end=9)
    assert unavailable.value.conflicts
    assert scheduler.snapshot() == before

    with pytest.raises(KeyError):
        scheduler.cancel("missing")
    cancelled = scheduler.cancel("first")
    assert not cancelled.reservation.active
    assert scheduler.cancel("first") == cancelled
    replay = scheduler.schedule(
        "first", "api", 2, earliest_start=0, latest_end=12, request_id="once"
    )
    assert replay == cancelled
    with pytest.raises(ValueError, match="cancelled"):
        scheduler.schedule(
            "dependent", "api", 1, dependencies=("first",), latest_end=12
        )


def test_radio_validation_conflict_query_and_domain_edges() -> None:
    with pytest.raises(ValueError, match="channel_count"):
        RadioSpectrumScheduler(0)
    scheduler = RadioSpectrumScheduler(10)
    with pytest.raises(ValueError, match="channel_start"):
        scheduler.reserve("owner", -1, 1, 0, 1)
    with pytest.raises(ValueError, match="channel_width"):
        scheduler.reserve("owner", 0, 0, 0, 1)
    with pytest.raises(ValueError, match="guard_channels"):
        scheduler.reserve("owner", 0, 1, 0, 1, guard_channels=-1)
    with pytest.raises(ValueError, match="managed channel"):
        scheduler.reserve("owner", 9, 2, 0, 1)
    with pytest.raises(ValueError, match="managed channel"):
        scheduler.conflicts_for(9, 2, 0, 1)

    first = scheduler.reserve("z", 0, 2, 5, 7, guard_channels=2)
    second = scheduler.reserve("a", 8, 2, 0, 2, guard_channels=2)
    assert first.active
    assert scheduler.conflicts_for(4, 2, 5, 6) is None
    conflict = scheduler.conflicts_for(2, 1, 5, 6, guard_channels=1)
    assert conflict is not None
    expected_conflicting_ids = (first.id,)
    assert conflict.conflicting_ids == expected_conflicting_ids
    with pytest.raises(SpectrumConflictError) as raised:
        scheduler.reserve("x", 2, 1, 5, 6, guard_channels=1)
    assert raised.value.conflict == conflict
    reservations = scheduler.snapshot().reservations
    expected_reservations = (second, first)
    assert reservations == expected_reservations


def test_radio_idempotency_cancellation_snapshot_and_factory() -> None:
    scheduler = RadioSpectrumScheduler(8)
    reservation = scheduler.reserve(
        "owner", 1, 2, 0, 2, request_id="once", guard_channels=1
    )
    assert (
        scheduler.reserve("owner", 1, 2, 0, 2, request_id="once", guard_channels=1)
        is reservation
    )
    before = scheduler.snapshot()
    with pytest.raises(ValueError, match="idempotency key"):
        scheduler.reserve("owner", 2, 2, 0, 2, request_id="once")
    with pytest.raises(KeyError):
        scheduler.cancel("owner", "missing:1")
    with pytest.raises(PermissionError):
        scheduler.cancel("other", reservation.id)
    assert scheduler.snapshot() == before

    cancelled = scheduler.cancel("owner", reservation.id)
    assert cancelled.status is SpectrumStatus.CANCELLED
    assert not cancelled.active
    assert scheduler.cancel("owner", reservation.id) is cancelled
    assert (
        scheduler.reserve("owner", 1, 2, 0, 2, request_id="once", guard_channels=1)
        is cancelled
    )
    assert scheduler.conflicts_for(1, 2, 0, 2, guard_channels=1) is None
    default_reservations = create_radio_spectrum_scheduler().snapshot().reservations
    assert not default_reservations


def test_render_public_validation_and_owned_assignment_edges() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        RenderWorker("worker", 0)
    with pytest.raises(ValueError, match="at least one resource"):
        RenderFarmScheduler(())
    worker = RenderWorker("worker")
    with pytest.raises(ValueError, match="resource names"):
        RenderFarmScheduler((worker, worker))

    scheduler = RenderFarmScheduler((worker,))
    with pytest.raises(ValueError, match="frame_count"):
        scheduler.assign_chunk("film", 0, 0, 1, earliest_start=0, latest_end=1)
    assignment = scheduler.assign_chunk(
        "film", 0, 5, 1, earliest_start=0, latest_end=2, request_id="assignment"
    )
    assert assignment.id == assignment.placement.id
    with pytest.raises(ValueError, match="different frame chunk"):
        scheduler.assign_chunk(
            "film", 10, 5, 1, earliest_start=0, latest_end=2, request_id="assignment"
        )
    with pytest.raises(KeyError):
        scheduler.cancel("film", "missing:1")
    with pytest.raises(PermissionError):
        scheduler.cancel("other", assignment.id)
    cancelled = scheduler.cancel("film", assignment.id)
    assert cancelled.status is ChunkStatus.CANCELLED
    assert scheduler.cancel("film", assignment.id) is cancelled
    with pytest.raises(ValueError, match="active assignment"):
        scheduler.retry(
            "film",
            assignment.id,
            duration=1,
            earliest_start=0,
            latest_end=2,
            request_id="retry",
        )
    default_assignments = create_render_farm_scheduler().snapshot().assignments
    assert not default_assignments


def test_render_retry_input_validation_is_failure_atomic() -> None:
    scheduler = RenderFarmScheduler((RenderWorker("worker"),))
    assignment = scheduler.assign_chunk("film", 0, 5, 1, earliest_start=0, latest_end=2)
    before = scheduler.snapshot()
    invalid_calls = (
        (0, 0, 2, "retry"),
        (2, 1, 2, "retry"),
        (1, 0, 2, ""),
    )
    for duration, earliest_start, latest_end, request_id in invalid_calls:
        with pytest.raises(ValueError):
            scheduler.retry(
                "film",
                assignment.id,
                duration=duration,
                earliest_start=earliest_start,
                latest_end=latest_end,
                request_id=request_id,
            )
        after = scheduler.snapshot()
        assert after == before

    with pytest.raises(KeyError):
        scheduler.retry(
            "film",
            "missing:1",
            duration=1,
            earliest_start=0,
            latest_end=2,
            request_id="r",
        )
    with pytest.raises(PermissionError):
        scheduler.retry(
            "other",
            assignment.id,
            duration=1,
            earliest_start=0,
            latest_end=2,
            request_id="r",
        )
