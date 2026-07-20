"""Contracts for deterministic cumulative resource reservations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from treemendous.applications._shared.capacity import CapacityVector
from treemendous.applications._shared.reservations import (
    IdempotencyEntry,
    Reservation,
    ReservationCheckpoint,
    ReservationConflict,
    ReservationConflictError,
    ReservationLedger,
    ReservationStatus,
    ResourceCapacity,
    ResourceRequirement,
)
from treemendous.domain import Span


def test_exact_reservations_use_half_open_cumulative_capacity() -> None:
    ledger = ReservationLedger({"node": CapacityVector(cpu=2, memory=8)})
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
        ledger.reserve_exact("owner", 0, 1, {"gpu": CapacityVector(slots=1)})
    with pytest.raises(ValueError, match="exceeds total"):
        ledger.reserve_exact(
            "owner",
            0,
            1,
            {"network": CapacityVector(bandwidth=11)},
        )
    with pytest.raises(KeyError, match="unknown"):
        ledger.reserve_exact("owner", 0, 1, {"missing": CapacityVector(units=1)})
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

    original = ledger.reserve_exact("alice", 1, 3, demand, request_id="request-1")
    replay = ledger.reserve_exact("alice", 1, 3, demand, request_id="request-1")
    assert replay is original

    with pytest.raises(ValueError, match="idempotency"):
        ledger.reserve_exact("alice", 3, 5, demand, request_id="request-1")

    other_owner = ledger.reserve_exact("bob", 3, 5, demand, request_id="request-1")
    assert original.id == "alice:1"
    assert other_owner.id == "bob:1"

    cancelled = ledger.cancel("alice", original.id)
    assert cancelled.status is ReservationStatus.CANCELLED
    assert ledger.cancel("alice", original.id) is cancelled
    assert (
        ledger.reserve_exact("alice", 1, 3, demand, request_id="request-1") is cancelled
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
    ledger.reserve_exact("existing", 0, 10, {"room": CapacityVector(units=1)})
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
    cancelled = ledger.reserve_exact("owner", -5, -1, demand, request_id="old")
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
    ledger.reserve_exact("owner", 0, 1, {"lane": CapacityVector(units=1)})
    before = ledger.checkpoint()
    invalid = replace(before, next_sequences=(("owner", 1),))

    with pytest.raises(ValueError, match="reuse"):
        ledger.restore(invalid)
    assert ledger.checkpoint() == before

    with pytest.raises(TypeError, match="ReservationCheckpoint"):
        ReservationLedger.from_checkpoint(object())  # type: ignore[arg-type]


def test_checkpoint_rejects_duplicate_owner_request_identity() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    demand = {"lane": CapacityVector(units=1)}
    ledger.reserve_exact("owner", 0, 1, demand, request_id="first")
    ledger.reserve_exact("owner", 1, 2, demand, request_id="second")
    checkpoint = ledger.checkpoint()
    duplicate = replace(
        checkpoint.reservations[1],
        request_id=checkpoint.reservations[0].request_id,
    )
    invalid = replace(
        checkpoint,
        reservations=(checkpoint.reservations[0], duplicate),
        idempotency=(checkpoint.idempotency[0],),
    )

    with pytest.raises(ValueError, match="duplicate owner/request identity"):
        ReservationLedger.from_checkpoint(invalid)


def test_checkpoint_rejects_missing_retained_reservation_history() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    demand = {"lane": CapacityVector(units=1)}
    for start in range(3):
        reservation = ledger.reserve_exact("owner", start, start + 1, demand)
        ledger.cancel("owner", reservation.id)
    checkpoint = ledger.checkpoint()
    invalid = replace(
        checkpoint,
        reservations=(checkpoint.reservations[0], checkpoint.reservations[2]),
    )

    with pytest.raises(ValueError, match="contiguous.*exact next counter"):
        ReservationLedger.from_checkpoint(invalid)


def test_checkpoint_rejects_non_exact_and_orphan_owner_counters() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    ledger.reserve_exact("owner", 0, 1, {"lane": CapacityVector(units=1)})
    checkpoint = ledger.checkpoint()
    non_exact = replace(checkpoint, next_sequences=(("owner", 10),))
    orphan = replace(
        checkpoint,
        next_sequences=checkpoint.next_sequences + (("other", 1),),
    )

    with pytest.raises(ValueError, match="exact next counter"):
        ReservationLedger.from_checkpoint(non_exact)
    with pytest.raises(ValueError, match="orphan counter"):
        ReservationLedger.from_checkpoint(orphan)


def test_diagnostics_reports_duplicate_request_identity() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    demand = {"lane": CapacityVector(units=1)}
    first = ledger.reserve_exact("owner", 0, 1, demand, request_id="first")
    second = ledger.reserve_exact("owner", 1, 2, demand, request_id="second")
    ledger._reservations[second.id] = replace(second, request_id="first")

    errors = ledger.diagnostics()
    assert "duplicate owner/request identity ('owner', 'first')" in errors
    assert (
        f"reservation {second.id!r} has an inconsistent idempotency identity" in errors
    )
    assert ledger.get(first.id).request_id == "first"


def test_diagnostics_reports_sequence_and_counter_corruption() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    demand = {"lane": CapacityVector(units=1)}
    for start in range(3):
        reservation = ledger.reserve_exact("owner", start, start + 1, demand)
        ledger.cancel("owner", reservation.id)
    del ledger._reservations["owner:2"]
    ledger._next_by_owner["owner"] = 8
    ledger._next_by_owner["orphan"] = 1

    errors = ledger.diagnostics()
    assert "orphan owner counter for 'orphan'" in errors
    assert "non-contiguous owner sequences or non-exact counter for 'owner'" in errors


def test_diagnostics_reports_inconsistent_idempotency_fingerprint() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    reservation = ledger.reserve_exact(
        "owner",
        0,
        1,
        {"lane": CapacityVector(units=1)},
        request_id="request",
    )
    entry = ledger.checkpoint().idempotency[0]
    invalid_fingerprint = replace(entry.fingerprint, start=1)
    ledger._idempotency[(entry.owner, entry.request_id)] = (
        invalid_fingerprint,
        reservation.id,
    )

    expected_errors = (
        f"reservation {reservation.id!r} has an inconsistent idempotency fingerprint",
    )
    assert ledger.diagnostics() == expected_errors


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


def test_public_reservation_value_validation_edges() -> None:
    capacity = CapacityVector(units=1)
    with pytest.raises(TypeError, match="resource must be a string"):
        ResourceRequirement(1, capacity)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resource must not be empty"):
        ResourceCapacity("", capacity)
    with pytest.raises(TypeError, match="capacity must be"):
        ResourceRequirement("lane", {"units": 1})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="capacity must be"):
        ResourceCapacity("lane", {"units": 1})  # type: ignore[arg-type]

    requirement = ResourceRequirement("lane", capacity)
    with pytest.raises(ValueError, match="at least one resource"):
        Reservation("owner:1", "owner", 0, 1, ())
    with pytest.raises(ValueError, match="unique resources"):
        Reservation("owner:1", "owner", 0, 1, (requirement, requirement))
    with pytest.raises(ValueError, match="request_id must not be empty"):
        Reservation("owner:1", "owner", 0, 1, (requirement,), request_id="")
    with pytest.raises(ValueError, match="buffer_before must be non-negative"):
        Reservation("owner:1", "owner", 0, 1, (requirement,), buffer_before=-1)
    with pytest.raises(TypeError, match="ReservationStatus"):
        Reservation("owner:1", "owner", 0, 1, (requirement,), status="active")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="IDs must be sorted"):
        ReservationConflict(
            "lane",
            0,
            1,
            capacity,
            capacity,
            capacity,
            ("z:1", "a:1"),
        )
    with pytest.raises(ValueError, match="same dimension"):
        ReservationConflict(
            "lane",
            0,
            1,
            capacity,
            CapacityVector(slots=1),
            capacity,
            (),
        )
    with pytest.raises(ValueError, match="requires conflict details"):
        ReservationConflictError(())


def test_public_ledger_request_validation_and_alias() -> None:
    with pytest.raises(TypeError, match="resources must be a mapping"):
        ReservationLedger([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one resource"):
        ReservationLedger({})
    with pytest.raises(TypeError, match="resource capacities"):
        ReservationLedger({"lane": 1})  # type: ignore[dict-item]

    ledger = ReservationLedger({"lane": {"units": 1}})
    demand = {"lane": {"units": 1}}
    with pytest.raises(TypeError, match="requirements must be"):
        ledger.reserve_exact("owner", 0, 1, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one resource requirement"):
        ledger.reserve_exact("owner", 0, 1, {})
    with pytest.raises(ValueError, match="owner must not be empty"):
        ledger.reserve_exact("", 0, 1, demand)
    with pytest.raises(ValueError, match="request_id must not be empty"):
        ledger.reserve_exact("owner", 0, 1, demand, request_id="")
    with pytest.raises(ValueError, match="buffer_after must be non-negative"):
        ledger.reserve_exact("owner", 0, 1, demand, buffer_after=-1)

    reservation = ledger.reserve(
        "owner", 0, 1, demand, buffer_before=1, buffer_after=2
    )
    assert reservation.occupied_span == Span(-1, 3)
    with pytest.raises(KeyError):
        ledger.get("missing:1")
    with pytest.raises(ValueError, match="reservation_id must not be empty"):
        ledger.get("")
    with pytest.raises(ValueError, match="owner must not be empty"):
        ledger.reservations(owner="")
    owner_reservations = ledger.reservations(owner="owner", active_only=True)
    expected_reservations = (reservation,)
    assert owner_reservations == expected_reservations


def test_earliest_validation_disjoint_history_and_idempotency() -> None:
    ledger = ReservationLedger(
        {"a": CapacityVector(units=1), "b": CapacityVector(units=1)}
    )
    demand_a = {"a": CapacityVector(units=1)}
    with pytest.raises(ValueError, match="search window"):
        ledger.reserve_earliest(
            "owner", 2, demand_a, earliest_start=2, latest_end=3
        )
    with pytest.raises(ValueError, match="buffer_before must be non-negative"):
        ledger.reserve_earliest("owner", 1, demand_a, buffer_before=-1)

    irrelevant = ledger.reserve_exact(
        "other", 0, 2, {"b": CapacityVector(units=1)}
    )
    cancelled = ledger.reserve_exact("old", 0, 2, demand_a)
    ledger.cancel("old", cancelled.id)
    reservation = ledger.reserve_earliest(
        "owner",
        2,
        demand_a,
        earliest_start=5,
        latest_end=8,
        request_id="earliest",
    )
    assert reservation.start == 5
    replay = ledger.reserve_earliest(
        "owner",
        2,
        demand_a,
        earliest_start=5,
        latest_end=8,
        request_id="earliest",
    )
    assert replay is reservation
    before = ledger.snapshot()
    with pytest.raises(ValueError, match="idempotency key"):
        ledger.reserve_earliest(
            "owner",
            1,
            demand_a,
            earliest_start=5,
            latest_end=8,
            request_id="earliest",
        )
    assert ledger.snapshot() == before
    assert ledger.get(irrelevant.id) is irrelevant
    ledger.assert_invariants()


def test_checkpoint_rejects_resource_identity_and_capacity_corruption() -> None:
    two_resources = ReservationLedger(
        {"a": CapacityVector(units=1), "b": CapacityVector(units=1)}
    ).checkpoint()
    invalid_resources = (
        replace(two_resources, resources=tuple(reversed(two_resources.resources))),
        replace(
            two_resources,
            resources=(two_resources.resources[0], two_resources.resources[0]),
        ),
    )
    for invalid in invalid_resources:
        with pytest.raises(ValueError, match="resources must be unique and sorted"):
            ReservationLedger.from_checkpoint(invalid)

    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    first = ledger.reserve_exact(
        "owner",
        0,
        2,
        {"lane": CapacityVector(units=1)},
        request_id="request",
    )
    checkpoint = ledger.checkpoint()
    duplicate = replace(checkpoint, reservations=(first, first))
    with pytest.raises(ValueError, match="duplicate reservation IDs"):
        ReservationLedger.from_checkpoint(duplicate)

    invalid_identities = (
        replace(first, reservation_id="other:1"),
        replace(first, reservation_id="owner:not-a-number"),
        replace(first, reservation_id="owner:0"),
        replace(first, reservation_id="owner:01"),
    )
    for invalid_reservation in invalid_identities:
        invalid = replace(checkpoint, reservations=(invalid_reservation,))
        with pytest.raises(ValueError, match="reservation ID"):
            ReservationLedger.from_checkpoint(invalid)

    other = replace(
        first,
        reservation_id="other:1",
        owner="other",
        request_id=None,
    )
    over_capacity = replace(
        checkpoint,
        reservations=(first, other),
        next_sequences=(("other", 2), ("owner", 2)),
    )
    with pytest.raises(ValueError, match="exceeds resource capacity"):
        ReservationLedger.from_checkpoint(over_capacity)


def test_checkpoint_rejects_counter_corruption_variants() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    ledger.reserve_exact("owner", 0, 1, {"lane": CapacityVector(units=1)})
    checkpoint = ledger.checkpoint()
    invalid_counters = (
        (checkpoint.next_sequences + checkpoint.next_sequences, "duplicate owner"),
        ((("owner", 0),), "positive"),
        ((), "omits the owner counter"),
    )
    for counters, message in invalid_counters:
        invalid = replace(checkpoint, next_sequences=counters)
        with pytest.raises(ValueError, match=message):
            ReservationLedger.from_checkpoint(invalid)


def test_checkpoint_rejects_idempotency_reference_corruption() -> None:
    ledger = ReservationLedger({"lane": CapacityVector(units=1)})
    ledger.reserve_exact(
        "owner",
        0,
        1,
        {"lane": CapacityVector(units=1)},
        request_id="request",
    )
    checkpoint = ledger.checkpoint()
    entry = checkpoint.idempotency[0]

    duplicate = replace(checkpoint, idempotency=(entry, entry))
    with pytest.raises(ValueError, match="duplicate idempotency keys"):
        ReservationLedger.from_checkpoint(duplicate)
    unknown = replace(
        checkpoint,
        idempotency=(replace(entry, reservation_id="missing:1"),),
    )
    with pytest.raises(ValueError, match="unknown reservation"):
        ReservationLedger.from_checkpoint(unknown)
    mismatched = replace(
        checkpoint,
        idempotency=(replace(entry, request_id="other-request"),),
    )
    with pytest.raises(ValueError, match="does not match its reservation"):
        ReservationLedger.from_checkpoint(mismatched)
    omitted = replace(checkpoint, idempotency=())
    with pytest.raises(ValueError, match="omits a reservation idempotency"):
        ReservationLedger.from_checkpoint(omitted)
    invalid_type = replace(
        checkpoint,
        idempotency=(
            IdempotencyEntry(entry.owner, entry.request_id, object(), entry.reservation_id),  # type: ignore[arg-type]
        ),
    )
    with pytest.raises(TypeError, match="fingerprint has an invalid type"):
        ReservationLedger.from_checkpoint(invalid_type)


def test_checkpoint_rejects_fingerprint_corruption_variants() -> None:
    exact = ReservationLedger({"lane": CapacityVector(units=1)})
    exact.reserve_exact(
        "owner",
        0,
        2,
        {"lane": CapacityVector(units=1)},
        request_id="exact",
    )
    checkpoint = exact.checkpoint()
    entry = checkpoint.idempotency[0]
    invalid_fingerprints = (
        (replace(entry.fingerprint, kind="unknown"), "invalid request kind"),
        (replace(entry.fingerprint, duration=1), "does not match"),
        (replace(entry.fingerprint, buffer_after=1), "does not match"),
        (replace(entry.fingerprint, start=1), "exact request fingerprint"),
        (replace(entry.fingerprint, latest_end=3), "exact request fingerprint"),
    )
    for fingerprint, message in invalid_fingerprints:
        invalid_entry = replace(entry, fingerprint=fingerprint)
        invalid = replace(checkpoint, idempotency=(invalid_entry,))
        with pytest.raises(ValueError, match=message):
            ReservationLedger.from_checkpoint(invalid)

    earliest = ReservationLedger({"lane": CapacityVector(units=1)})
    earliest.reserve_earliest(
        "owner",
        2,
        {"lane": CapacityVector(units=1)},
        earliest_start=0,
        latest_end=4,
        request_id="earliest",
    )
    earliest_checkpoint = earliest.checkpoint()
    earliest_entry = earliest_checkpoint.idempotency[0]
    too_late = replace(earliest_entry.fingerprint, start=1)
    invalid = replace(
        earliest_checkpoint,
        idempotency=(replace(earliest_entry, fingerprint=too_late),),
    )
    with pytest.raises(ValueError, match="earliest request fingerprint"):
        ReservationLedger.from_checkpoint(invalid)


def test_successful_restore_replaces_state_and_preserves_diagnostics() -> None:
    source = ReservationLedger({"source": CapacityVector(units=1)})
    reservation = source.reserve_exact(
        "owner",
        2,
        4,
        {"source": CapacityVector(units=1)},
        request_id="request",
    )
    target = ReservationLedger({"target": CapacityVector(units=2)})
    target.reserve_exact("old", 0, 1, {"target": CapacityVector(units=1)})

    target.restore(source.checkpoint())

    expected_snapshot = source.snapshot()
    assert target.snapshot() == expected_snapshot
    assert target.get(reservation.id) == reservation
    assert not target.diagnostics()
    target.assert_invariants()
