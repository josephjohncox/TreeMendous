"""GPU stream scheduler with device/stream compatibility and dependency readiness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import (
    Reservation,
    ReservationConflict,
    ReservationLedger,
    ReservationSnapshot,
)
from treemendous.applications.scheduling._common import (
    SchedulingUnavailableError,
    names,
    positive,
    text,
)
from treemendous.domain import validate_coordinate


@dataclass(frozen=True)
class GPUStream:
    """A named stream accepting kernel compatibility labels."""

    name: str
    compatibility: frozenset[str]

    def __post_init__(self) -> None:
        text(self.name, "stream name")
        names(self.compatibility, "stream compatibility")


@dataclass(frozen=True)
class GPUDevice:
    """A device with cumulative capacities and explicit streams."""

    name: str
    capacity: CapacityVector
    compatibility: frozenset[str]
    streams: tuple[GPUStream, ...]

    def __post_init__(self) -> None:
        text(self.name, "device name")
        if not isinstance(self.capacity, CapacityVector):
            raise TypeError("capacity must be a CapacityVector")
        names(self.compatibility, "device compatibility")
        if not self.streams:
            raise ValueError("a GPU device must expose at least one stream")
        if len({stream.name for stream in self.streams}) != len(self.streams):
            raise ValueError("stream names must be unique within a device")


@dataclass(frozen=True)
class GPUPlacement:
    """Committed device and stream reservation for one kernel."""

    device: str
    stream: str
    reservation: Reservation

    @property
    def start(self) -> int:
        return self.reservation.start

    @property
    def end(self) -> int:
        return self.reservation.end

    @property
    def id(self) -> str:
        return self.reservation.id


class GPUStreamScheduler:
    """Bounded deterministic stream placement; not a GPU execution runtime."""

    def __init__(self, devices: tuple[GPUDevice, ...]) -> None:
        if not devices:
            raise ValueError("at least one GPU device is required")
        self._devices = {device.name: device for device in devices}
        if len(self._devices) != len(devices):
            raise ValueError("GPU device names must be unique")
        self._device_resources: dict[str, str] = {}
        self._stream_resources: dict[tuple[str, str], str] = {}
        resources: dict[str, CapacityVector] = {}
        for device_index, device in enumerate(
            sorted(devices, key=lambda item: item.name)
        ):
            device_resource = f"device:{device_index}"
            self._device_resources[device.name] = device_resource
            resources[device_resource] = device.capacity
            for stream_index, stream in enumerate(
                sorted(device.streams, key=lambda item: item.name)
            ):
                stream_resource = f"stream:{device_index}:{stream_index}"
                self._stream_resources[(device.name, stream.name)] = stream_resource
                resources[stream_resource] = CapacityVector(units=1)
        self._ledger = ReservationLedger(resources)
        self._placements: dict[str, GPUPlacement] = {}
        self._requests: dict[tuple[str, str], tuple[object, GPUPlacement]] = {}
        self._lock = RLock()

    def schedule(
        self,
        kernel_id: str,
        duration: int,
        demand: CapacityVector,
        *,
        compatibility: str,
        dependency_ready_times: Mapping[str, int] | None = None,
        earliest_start: int = 0,
        latest_end: int,
        request_id: str | None = None,
    ) -> GPUPlacement:
        text(kernel_id, "kernel_id")
        positive(duration, "duration")
        if not isinstance(demand, CapacityVector):
            raise TypeError("demand must be a CapacityVector")
        text(compatibility, "compatibility")
        validate_coordinate(earliest_start, "earliest_start")
        validate_coordinate(latest_end, "latest_end")
        dependencies = dependency_ready_times or {}
        for dependency, ready in dependencies.items():
            text(dependency, "dependency")
            validate_coordinate(ready, "dependency ready time")
        ready_start = max((earliest_start, *dependencies.values()))
        if ready_start + duration > latest_end:
            raise SchedulingUnavailableError("dependencies leave no feasible window")
        if request_id is not None:
            text(request_id, "request_id")
        fingerprint: object = (
            duration,
            demand,
            compatibility,
            tuple(sorted(dependencies.items())),
            earliest_start,
            latest_end,
        )
        with self._lock:
            if request_id is not None:
                prior = self._requests.get((kernel_id, request_id))
                if prior is not None:
                    if prior[0] != fingerprint:
                        raise ValueError(
                            "idempotency key was already used for a different request"
                        )
                    return prior[1]

            pairs: list[tuple[str, str]] = []
            for device in sorted(self._devices.values(), key=lambda item: item.name):
                if compatibility not in device.compatibility:
                    continue
                try:
                    device.capacity._require_same_dimensions(demand)
                except ValueError:
                    continue
                if not device.capacity.fits(demand):
                    continue
                for stream in sorted(device.streams, key=lambda item: item.name):
                    if compatibility in stream.compatibility:
                        pairs.append((device.name, stream.name))
            if not pairs:
                raise SchedulingUnavailableError(
                    "no compatible device/stream pair satisfies total demand"
                )

            selected: tuple[int, str, str] | None = None
            last_conflicts: tuple[ReservationConflict, ...] = ()
            for start in range(ready_start, latest_end - duration + 1):
                for device_name, stream_name in pairs:
                    requirements = {
                        self._device_resources[device_name]: demand,
                        self._stream_resources[(device_name, stream_name)]: (
                            CapacityVector(units=1)
                        ),
                    }
                    conflicts = self._ledger.conflicts_for(
                        start, start + duration, requirements
                    )
                    if not conflicts:
                        selected = start, device_name, stream_name
                        break
                    last_conflicts = conflicts
                if selected is not None:
                    break
            if selected is None:
                raise SchedulingUnavailableError(
                    "compatible GPU streams have no bounded capacity",
                    conflicts=last_conflicts,
                )
            start, device_name, stream_name = selected
            requirements = {
                self._device_resources[device_name]: demand,
                self._stream_resources[(device_name, stream_name)]: (
                    CapacityVector(units=1)
                ),
            }
            reservation = self._ledger.reserve_exact(
                kernel_id,
                start,
                start + duration,
                requirements,
                request_id=request_id,
            )
            placement = GPUPlacement(device_name, stream_name, reservation)
            self._placements[placement.id] = placement
            if request_id is not None:
                self._requests[(kernel_id, request_id)] = fingerprint, placement
            return placement

    def cancel(self, kernel_id: str, reservation_id: str) -> GPUPlacement:
        with self._lock:
            try:
                prior_placement = self._placements[reservation_id]
            except KeyError:
                raise KeyError(reservation_id) from None
            reservation = self._ledger.cancel(kernel_id, reservation_id)
            placement = GPUPlacement(
                prior_placement.device,
                prior_placement.stream,
                reservation,
            )
            self._placements[reservation_id] = placement
            for key, (fingerprint, prior) in tuple(self._requests.items()):
                if prior.id == reservation_id:
                    self._requests[key] = fingerprint, placement
            return placement

    def snapshot(self) -> ReservationSnapshot:
        return self._ledger.snapshot()


def create_gpu_stream_scheduler(
    *, devices: tuple[GPUDevice, ...] | None = None
) -> GPUStreamScheduler:
    """Construct a GPU stream scheduler."""
    return GPUStreamScheduler(
        devices
        or (
            GPUDevice(
                "gpu-a",
                CapacityVector(memory=16, slots=2),
                frozenset({"compute"}),
                (GPUStream("stream-0", frozenset({"compute"})),),
            ),
        )
    )
