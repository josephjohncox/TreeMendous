"""Contracts for deterministic cumulative resource reservations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import (
    ReservationCheckpoint,
    ReservationConflictError,
    ReservationLedger,
    ReservationStatus,
)


def test_exact_reservations_use_half_open_cumulative_capacity() -> None:
    ledger = ReservationLedger(
        {"node": CapacityVector(cpu=2, memory=8)}
    )
    one = {"node": CapacityVector(cpu=1, memory=4)}

    first = ledger.reserve_exact("alpha", 10, 20, one)
    second = ledger.reserve_exact("beta", 10, 20, one)
    touching = ledger.reserve_exact("gamma", 20, 25, one)

    assert first.occupied_span.end == touching.occupied_span.start
    assert second.start == first.start
    with pytest.raises(ReservationConflictError) as raised:
        ledger.reserve_exact("delta", 15, 16, one)

    conflict = raised.value.conflicts[0]
    assert conflict.resource == "node"
    assert conflict.start == 15
    assert conflict.end == 16
    assert conflict.used == CapacityVector(cpu=2, memory=8)
    assert conflict.requested == CapacityVector(cpu=1, memory=4)
    expected_ids = tuple(sorted([first.id, second.id]))
    assert conflict.reservation_ids == expected_ids


def test_capacity_dimensions_are_checked_per_named_resource() -> None:
    ledger = ReservationLedger(
        {
            "gpu": CapacityVector(slots=2, memory=16),
            "network": CapacityVector(bandwidth=10),
        }
    )

    with pytest.raises(ValueError, match="same dimension"):
        ledger.reserve_exact(
            "owner", 0, 1, {"gpu": CapacityVector(slots=1)}
        )
    with pytest.raises(ValueError, match="exceeds total"):
        ledger.reserve_exact(
            "owner",
            0,
            1,
            {"network": CapacityVector(bandwidth=11)},
        )
    with pytest.raises(KeyError, match="unknown"):
        ledger.reserve_exact(
            "owner", 0, 1, {"missing": CapacityVector(units=1)}
        )
    assert not ledger.reservations()


def test_earliest_reservation_is_deterministic_with_buffers() -> None:
    ledger = ReservationLedger({"gate": CapacityVector(units=1)})
    demand = {"gate": CapacityVector(units=1)}
    first = ledger.reserve_exact(
        "flight-a",
        10,
        20,
        demand,
        buffer_before=2,
        buffer_after=3,
    )

    before = ledger.reserve_earliest(
        "flight-b",
        3,
        demand,
        earliest_start=0,
        latest_end=8,
        buffer_after=2,
    )
    after = ledger.reserve_earliest(
        "flight-c",
        4,
        demand,
        earliest_start=10,
        buffer_before=2,
    )

    assert first.occupied_span.start == 8
    assert first.occupied_span.end == 23
    assert before.span.start == 0
    assert before.occupied_span.end == 5
    assert after.start == 25
    assert after.occupied_span.start == first.occupied_span.end

    with pytest.raises(ReservationConflictError):
        ledger.reserve_earliest(
            "flight-d",
            4,
            demand,
            earliest_start=10,
            latest_end=24,
        )


def test_owner_scoped_idempotency_preserves_identity_and_rejects_reuse() -> None:
    ledger = ReservationLedger({"room": CapacityVector(units=1)})
    demand = {"room": CapacityVector(units=1)}

    original = ledger.reserve_exact(
        "alice", 1, 3, demand, request_id="request-1"
    )
    replay = ledger.reserve_exact(
        "alice", 1, 3, demand, request_id="request-1"
    )
    assert replay is original

    with pytest.raises(ValueError, match="idempotency"):
        ledger.reserve_exact(
            "alice", 3, 5, demand, request_id="request-1"
        )

    other_owner = ledger.reserve_exact(
        "bob", 3, 5, demand, request_id="request-1"
    )
    assert original.id == "alice:1"
    assert other_owner.id == "bob:1"

    cancelled = ledger.cancel("alice", original.id)
    assert cancelled.status is ReservationStatus.CANCELLED
    assert ledger.cancel("alice", original.id) is cancelled
    assert (
        ledger.reserve_exact(
            "alice", 1, 3, demand, request_id="request-1"
        )
        is cancelled
    )
    ledger.reserve_exact("carol", 1, 3, demand)

    with pytest.raises(PermissionError, match="different owner"):
        ledger.cancel("bob", original.id)
    with pytest.raises(KeyError):
        ledger.cancel("alice", "alice:999")


def test_multi_resource_failure_is_atomic_before_exposure() -> None:
    ledger = ReservationLedger(
        {
            "room": CapacityVector(units=1),
            "projector": CapacityVector(units=1),
        }
    )
    ledger.reserve_exact(
        "existing", 0, 10, {"room": CapacityVector(units=1)}
    )
    before = ledger.snapshot()

    with pytest.raises(ReservationConflictError):
        ledger.reserve_exact(
            "candidate",
            5,
            8,
            {
                "projector": CapacityVector(units=1),
                "room": CapacityVector(units=1),
            },
            request_id="atomic",
        )

    assert ledger.snapshot() == before
    projector = ledger.reserve_exact(
        "other", 5, 8, {"projector": CapacityVector(units=1)}
    )
    assert projector.id == "other:1"
    candidate = ledger.reserve_exact(
        "candidate",
        10,
        12,
        {
            "room": CapacityVector(units=1),
            "projector": CapacityVector(units=1),
        },
        request_id="atomic",
    )
    assert candidate.id == "candidate:1"


def test_conflict_query_and_snapshot_are_deterministic_and_non_mutating() -> None:
    ledger = ReservationLedger({"worker": CapacityVector(slots=2)})
    demand = {"worker": CapacityVector(slots=1)}
    later = ledger.reserve_exact("z", 5, 10, demand)
    earlier = ledger.reserve_exact("a", 0, 4, demand)
    snapshot = ledger.snapshot()

    conflicts = ledger.conflicts_for(3, 7, {"worker": CapacityVector(slots=2)})
    assert len(conflicts) == 2
    assert conflicts[0].start == 3
    assert conflicts[0].end == 4
    assert conflicts[1].start == 5
    assert conflicts[1].end == 7
    assert ledger.snapshot() == snapshot
    expected_reservations = tuple([earlier, later])
    assert snapshot.reservations == expected_reservations
    assert not ledger.diagnostics()


def test_checkpoint_restore_preserves_counters_cancellation_and_idempotency() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    demand = {"lane": CapacityVector(units=1)}
    cancelled = ledger.reserve_exact(
        "owner", -5, -1, demand, request_id="old"
    )
    ledger.cancel("owner", cancelled.id)
    active = ledger.reserve_earliest(
        "owner",
        3,
        demand,
        earliest_start=0,
        latest_end=10,
        request_id="new",
    )

    checkpoint = ledger.checkpoint()
    restored = ReservationLedger.from_checkpoint(checkpoint)
    assert restored.checkpoint() == checkpoint
    assert restored.snapshot() == ledger.snapshot()
    assert (
        restored.reserve_earliest(
            "owner",
            3,
            demand,
            earliest_start=0,
            latest_end=10,
            request_id="new",
        )
        == active
    )
    next_reservation = restored.reserve_exact("owner", 10, 11, demand)
    assert next_reservation.id == "owner:3"
    restored.assert_invariants()


def test_restore_rejects_invalid_checkpoint_without_changing_live_state() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    ledger.reserve_exact(
        "owner", 0, 1, {"lane": CapacityVector(units=1)}
    )
    before = ledger.checkpoint()
    invalid = replace(before, next_sequences=(("owner", 1),))

    with pytest.raises(ValueError, match="reuse"):
        ledger.restore(invalid)
    assert ledger.checkpoint() == before

    with pytest.raises(TypeError, match="ReservationCheckpoint"):
        ReservationLedger.from_checkpoint(object())  # type: ignore[arg-type]


def test_concurrent_earliest_reservations_are_serialized() -> None:
    ledger = ReservationLedger({"runner": CapacityVector(units=1)})
    demand = {"runner": CapacityVector(units=1)}

    def reserve(index: int) -> tuple[int, str]:
        reservation = ledger.reserve_earliest(
            f"worker-{index}",
            1,
            demand,
            earliest_start=0,
            request_id="job",
        )
        return reservation.start, reservation.id

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(reserve, range(40)))

    assert sorted(start for start, _ in results) == list(range(40))
    assert len({reservation_id for _, reservation_id in results}) == 40
    assert len(ledger.reservations(active_only=True)) == 40
    assert not ledger.diagnostics()


def test_checkpoint_type_is_part_of_exact_restore_contract() -> None:
    ledger = ReservationLedger({"resource": CapacityVector(units=1)})
    checkpoint = ledger.checkpoint()
    assert isinstance(checkpoint, ReservationCheckpoint)
