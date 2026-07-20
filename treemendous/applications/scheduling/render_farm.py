"""Render-farm frame chunk allocation, cancellation, and retry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import ReservationSnapshot
from treemendous.applications.scheduling._common import (
    Placement,
    integer,
    positive,
    text,
)
from treemendous.applications.scheduling._placement import (
    BoundedPlacementEngine,
    LabeledResource,
)
from treemendous.domain import Span, validate_coordinate


@dataclass(frozen=True)
class RenderWorker:
    """A worker with positive concurrent render slots."""

    name: str
    concurrency: int = 1

    def __post_init__(self) -> None:
        text(self.name, "worker name")
        integer(self.concurrency, "concurrency", minimum=1)


class ChunkStatus(str, Enum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    RETRIED = "retried"


@dataclass(frozen=True)
class FrameChunkAssignment:
    """One contiguous frame chunk assigned to a worker-time reservation."""

    render_id: str
    frames: Span
    attempt: int
    placement: Placement
    status: ChunkStatus = ChunkStatus.ACTIVE

    @property
    def id(self) -> str:
        return self.placement.id


@dataclass(frozen=True)
class RenderFarmSnapshot:
    """Immutable assignment history and underlying worker schedule."""

    assignments: tuple[FrameChunkAssignment, ...]
    reservations: ReservationSnapshot


class RenderFarmScheduler:
    """Allocates exact frame chunks; it does not predict render duration."""

    def __init__(self, workers: tuple[RenderWorker, ...]) -> None:
        self._engine = BoundedPlacementEngine(
            tuple(
                LabeledResource(
                    worker.name,
                    CapacityVector(slots=worker.concurrency),
                )
                for worker in workers
            )
        )
        self._assignments: dict[str, FrameChunkAssignment] = {}
        self._retry_requests: dict[
            tuple[str, str], tuple[object, FrameChunkAssignment]
        ] = {}
        self._lock = RLock()

    def assign_chunk(
        self,
        render_id: str,
        frame_start: int,
        frame_count: int,
        duration: int,
        *,
        earliest_start: int,
        latest_end: int,
        request_id: str | None = None,
        _replacing: str | None = None,
    ) -> FrameChunkAssignment:
        text(render_id, "render_id")
        integer(frame_start, "frame_start")
        integer(frame_count, "frame_count", minimum=1)
        frames = Span(frame_start, frame_start + frame_count)
        with self._lock:
            for assignment in self._assignments.values():
                if (
                    assignment.id != _replacing
                    and assignment.render_id == render_id
                    and assignment.status is ChunkStatus.ACTIVE
                    and assignment.frames.overlaps(frames)
                ):
                    raise ValueError(
                        f"frame chunk overlaps active assignment {assignment.id!r}"
                    )
            placement = self._engine.place(
                render_id,
                duration,
                CapacityVector(slots=1),
                earliest_start=earliest_start,
                latest_end=latest_end,
                request_id=request_id,
            )
            prior = self._assignments.get(placement.id)
            if prior is not None:
                if prior.frames != frames:
                    raise ValueError(
                        "idempotency key was already used for a different frame chunk"
                    )
                return prior
            attempt = 1
            if _replacing is not None:
                attempt = self._assignments[_replacing].attempt + 1
            assignment = FrameChunkAssignment(render_id, frames, attempt, placement)
            self._assignments[placement.id] = assignment
            return assignment

    def retry(
        self,
        render_id: str,
        assignment_id: str,
        *,
        duration: int,
        earliest_start: int,
        latest_end: int,
        request_id: str,
    ) -> FrameChunkAssignment:
        """Atomically schedule a replacement before retiring the failed attempt."""
        text(render_id, "render_id")
        text(assignment_id, "assignment_id")
        positive(duration, "duration")
        validate_coordinate(earliest_start, "earliest_start")
        validate_coordinate(latest_end, "latest_end")
        if earliest_start + duration > latest_end:
            raise ValueError("duration does not fit within the bounded window")
        text(request_id, "request_id")
        fingerprint: object = (
            assignment_id,
            duration,
            earliest_start,
            latest_end,
        )
        with self._lock:
            request_key = (render_id, request_id)
            prior_request = self._retry_requests.get(request_key)
            if prior_request is not None:
                if prior_request[0] != fingerprint:
                    raise ValueError(
                        "idempotency key was already used for a different retry"
                    )
                return prior_request[1]

            previous = self._owned_active(render_id, assignment_id)
            replacement = self.assign_chunk(
                render_id,
                previous.frames.start,
                previous.frames.length,
                duration,
                earliest_start=earliest_start,
                latest_end=latest_end,
                request_id=request_id,
                _replacing=assignment_id,
            )
            if (
                replacement.id == assignment_id
                or replacement.status is not ChunkStatus.ACTIVE
                or not replacement.placement.reservation.active
            ):
                raise ValueError(
                    "retry request must create a distinct active replacement assignment"
                )
            cancelled_placement = self._engine.cancel(render_id, assignment_id)
            retired = replace(
                previous,
                placement=cancelled_placement,
                status=ChunkStatus.RETRIED,
            )
            self._record_assignment(retired)
            self._retry_requests[request_key] = fingerprint, replacement
            return replacement

    def cancel(self, render_id: str, assignment_id: str) -> FrameChunkAssignment:
        with self._lock:
            assignment = self._assignments.get(assignment_id)
            if assignment is None:
                raise KeyError(assignment_id)
            if assignment.render_id != render_id:
                raise PermissionError("assignment belongs to a different render")
            if assignment.status is not ChunkStatus.ACTIVE:
                return assignment
            cancelled_placement = self._engine.cancel(render_id, assignment_id)
            cancelled = replace(
                assignment,
                placement=cancelled_placement,
                status=ChunkStatus.CANCELLED,
            )
            self._record_assignment(cancelled)
            return cancelled

    def _record_assignment(self, assignment: FrameChunkAssignment) -> None:
        self._assignments[assignment.id] = assignment
        for key, (fingerprint, prior) in tuple(self._retry_requests.items()):
            if prior.id == assignment.id:
                self._retry_requests[key] = fingerprint, assignment

    def _owned_active(self, render_id: str, assignment_id: str) -> FrameChunkAssignment:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        if assignment.render_id != render_id:
            raise PermissionError("assignment belongs to a different render")
        if assignment.status is not ChunkStatus.ACTIVE:
            raise ValueError("only an active assignment can be retried")
        return assignment

    def snapshot(self) -> RenderFarmSnapshot:
        with self._lock:
            assignments = tuple(
                sorted(
                    self._assignments.values(),
                    key=lambda item: (item.frames.start, item.id),
                )
            )
            return RenderFarmSnapshot(assignments, self._engine.snapshot())


def create_render_farm_scheduler(
    *, workers: tuple[RenderWorker, ...] | None = None
) -> RenderFarmScheduler:
    """Construct a render-farm scheduler."""
    return RenderFarmScheduler(workers or (RenderWorker("worker-a"),))
