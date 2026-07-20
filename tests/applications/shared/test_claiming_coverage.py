"""Public adversarial coverage for claim validation and checkpoint restore."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from treemendous.applications._shared.claiming import (
    ClaimInvariantError,
    ClaimLedger,
    ClaimState,
    ClaimUnavailableError,
    ExpiredClaimError,
    ForeignClaimError,
    WorkClaim,
)
from treemendous.applications._shared.clock import LogicalClock
from treemendous.domain import Span


class _EqualCoordinate(int):
    """An int subclass that compares equal but fails exact-type validation."""


def _replace_event(
    checkpoint: Any,
    sequence: int,
    *,
    payload_changes: dict[str, Any] | None = None,
    **event_changes: Any,
) -> Any:
    """Alter serialized evidence while keeping the event-log envelope coherent."""
    events = list(checkpoint.events.events)
    event = events[sequence - 1]
    if payload_changes is not None:
        payload = dict(event.payload)
        payload.update(payload_changes)
        event_changes["payload"] = tuple(payload.items())
    changed = replace(event, **event_changes)
    events[sequence - 1] = changed

    requests = list(checkpoint.events.requests)
    request_index = next(
        index
        for index, request in enumerate(requests)
        if request.event_sequence == sequence
    )
    requests[request_index] = replace(
        requests[request_index],
        stream=changed.stream,
        key=changed.idempotency_key,
        kind=changed.kind,
        payload=changed.payload,
        occurred_at=changed.occurred_at,
    )
    return replace(
        checkpoint,
        events=replace(
            checkpoint.events,
            events=tuple(events),
            requests=tuple(requests),
        ),
    )


def _restore(checkpoint: Any) -> ClaimLedger:
    return ClaimLedger.from_checkpoint(
        checkpoint, clock=LogicalClock(checkpoint.last_observed_at)
    )


def test_public_claim_values_requests_and_ledger_properties_validate() -> None:
    with pytest.raises(TypeError, match="ledger_id"):
        WorkClaim(1, 1, "owner", Span(0, 1), 1, 0, None)  # type: ignore[arg-type]

    active = WorkClaim("ledger", 1, "owner", Span(0, 1), 1, 0, None)
    terminal = replace(active, state=ClaimState.ABANDONED, revision=2)
    assert active.active
    assert not terminal.active

    ledger = ClaimLedger((0, 5), clock=LogicalClock())
    assert ledger.ledger_id
    assert ledger.domain.measure == 5
    bounded = ledger.claim_next(
        "worker",
        1,
        not_before=1,
        not_after=4,
        ttl=2,
        request_id="bounded",
    )
    checkpoint = ledger.checkpoint()
    request = checkpoint.requests[0]
    assert bounded.span == Span(1, 2)
    with pytest.raises(ValueError, match="not_after"):
        replace(request, not_after=request.not_before)
    with pytest.raises(ValueError, match="claim_id"):
        replace(request, claim_id=0)


def test_public_handle_validation_and_due_transitions_are_fenced() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 4), clock=clock)
    claim = ledger.claim_next("owner", 1, ttl=1)
    assert ledger.validate_active(claim) == claim

    forged_owner = replace(claim, owner="forged")
    with pytest.raises(ForeignClaimError, match="owner"):
        ledger.validate_active(forged_owner)

    clock.advance()
    with pytest.raises(ExpiredClaimError):
        ledger.complete(claim)
    assert ledger.snapshot().claims[0].state is ClaimState.EXPIRED


def test_unavailable_bounded_claim_still_commits_unrelated_expiry() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 3), clock=clock)
    completed = ledger.claim_next("finished", 1)
    ledger.complete(completed)
    expiring = ledger.claim_next("expiring", 1, ttl=1)
    clock.advance()

    with pytest.raises(ClaimUnavailableError):
        ledger.claim_next("blocked", 1, not_before=0, not_after=1)

    snapshot = ledger.snapshot()
    assert snapshot.claims[1].state is ClaimState.EXPIRED
    expected_available = (Span(1, 3),)
    assert snapshot.available == expected_available
    assert expiring.span == Span(1, 2)


def test_restore_rejects_claim_event_envelope_corruption() -> None:
    ledger = ClaimLedger((0, 2), clock=LogicalClock())
    ledger.claim_next("owner", 1)
    checkpoint = ledger.checkpoint()
    event = checkpoint.events.events[0]

    missing_claim_stream = _replace_event(checkpoint, 1, stream="claim:99")
    wrong_revision_count = replace(
        checkpoint,
        claims=(replace(checkpoint.claims[0], revision=2),),
    )
    extra_payload = _replace_event(checkpoint, 1, payload_changes={"extra": 1})
    wrong_revision = _replace_event(checkpoint, 1, payload_changes={"revision": 2})
    invalid_coordinate = _replace_event(
        checkpoint,
        1,
        payload_changes={"claim_id": _EqualCoordinate(1)},
    )
    invalid_initial_kind = _replace_event(
        checkpoint,
        1,
        kind="completed",
        idempotency_key="1:completed",
    )
    late_event = _replace_event(checkpoint, 1, occurred_at=1)
    mismatched_acquisition_time = replace(
        _replace_event(checkpoint, 1, occurred_at=1), last_observed_at=1
    )
    active_result = _replace_event(
        checkpoint,
        1,
        payload_changes={"result": {"unexpected": 1}},
    )

    corruptions = (
        (
            replace(checkpoint, claims=(object(),)),  # type: ignore[arg-type]
            TypeError,
            "WorkClaim",
        ),
        (missing_claim_stream, ClaimInvariantError, "streams"),
        (wrong_revision_count, ClaimInvariantError, "count"),
        (extra_payload, ClaimInvariantError, "fields"),
        (wrong_revision, ClaimInvariantError, "revision"),
        (invalid_coordinate, ClaimInvariantError, "coordinates"),
        (invalid_initial_kind, ClaimInvariantError, "begin"),
        (late_event, ClaimInvariantError, "after checkpoint"),
        (mismatched_acquisition_time, ClaimInvariantError, "timestamp"),
        (active_result, ClaimInvariantError, "result"),
    )
    assert event.kind == "active"
    for corrupted, error, message in corruptions:
        with pytest.raises(error, match=message):
            _restore(corrupted)


def test_restore_rejects_invalid_acquisition_expiry_evidence() -> None:
    ledger = ClaimLedger((0, 1), clock=LogicalClock())
    ledger.claim_next("owner", 1, ttl=2)
    checkpoint = ledger.checkpoint()

    invalid_expiry_type = _replace_event(
        checkpoint,
        1,
        payload_changes={"expires_at": _EqualCoordinate(2)},
    )
    expired_on_acquisition = _replace_event(
        checkpoint, 1, payload_changes={"expires_at": 0}
    )
    for corrupted, message in (
        (invalid_expiry_type, "expiry"),
        (expired_on_acquisition, "acquisition expiry"),
    ):
        with pytest.raises(ClaimInvariantError, match=message):
            _restore(corrupted)


def test_restore_rejects_corrupted_renewal_evidence() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 1), clock=clock)
    claim = ledger.claim_next("owner", 1, ttl=2)
    ledger.renew(claim, ttl=3)
    checkpoint = ledger.checkpoint()
    first_payload = dict(checkpoint.events.events[0].payload)

    reused_token = _replace_event(
        checkpoint,
        2,
        payload_changes={"fencing_token": first_payload["fencing_token"]},
    )
    missing_expiry = _replace_event(checkpoint, 2, payload_changes={"expires_at": None})
    renewed_after_expiry = replace(
        _replace_event(checkpoint, 2, occurred_at=2), last_observed_at=2
    )
    renewal_result = _replace_event(
        checkpoint,
        2,
        payload_changes={"result": {"unexpected": 1}},
    )

    for corrupted, message in (
        (reused_token, "fencing token"),
        (missing_expiry, "renewal expiry"),
        (renewed_after_expiry, "after expiry"),
        (renewal_result, "result"),
    ):
        with pytest.raises(ClaimInvariantError, match=message):
            _restore(corrupted)


def test_restore_rejects_corrupted_terminal_evidence() -> None:
    completed_ledger = ClaimLedger((0, 1), clock=LogicalClock())
    completed_claim = completed_ledger.claim_next("owner", 1, ttl=5)
    completed_ledger.complete(completed_claim)
    completed = completed_ledger.checkpoint()
    changed_lease = _replace_event(
        completed,
        2,
        payload_changes={"fencing_token": completed_claim.fencing_token + 1},
    )
    invalid_state = _replace_event(
        completed,
        2,
        kind="unknown",
        idempotency_key="2:unknown",
    )
    mismatched_final = replace(
        completed,
        claims=(replace(completed.claims[0], state=ClaimState.ABANDONED),),
    )

    abandoned_ledger = ClaimLedger((0, 1), clock=LogicalClock())
    abandoned_ledger.abandon(abandoned_ledger.claim_next("owner", 1))
    abandoned = abandoned_ledger.checkpoint()
    abandoned_result = _replace_event(
        abandoned,
        2,
        payload_changes={"result": {"unexpected": 1}},
    )

    for corrupted, message in (
        (changed_lease, "lease evidence"),
        (invalid_state, "state"),
        (mismatched_final, "final"),
        (abandoned_result, "result"),
    ):
        with pytest.raises(ClaimInvariantError, match=message):
            _restore(corrupted)


def test_restore_rejects_corrupted_expiration_evidence() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 1), clock=clock)
    claim = ledger.claim_next("owner", 1, ttl=1)
    clock.advance()
    ledger.expire()
    checkpoint = ledger.checkpoint()

    changed_lease = _replace_event(
        checkpoint,
        2,
        payload_changes={"fencing_token": claim.fencing_token + 1},
    )
    early_expiration = replace(
        _replace_event(checkpoint, 2, occurred_at=0), last_observed_at=1
    )
    expiration_result = _replace_event(
        checkpoint,
        2,
        payload_changes={"result": {"unexpected": 1}},
    )

    for corrupted, message in (
        (changed_lease, "lease evidence"),
        (early_expiration, "before lease expiry"),
        (expiration_result, "result"),
    ):
        with pytest.raises(ClaimInvariantError, match=message):
            _restore(corrupted)


def test_restore_rejects_expired_active_and_skipped_fencing_counter() -> None:
    ledger = ClaimLedger((0, 1), clock=LogicalClock())
    ledger.claim_next("owner", 1, ttl=1)
    checkpoint = ledger.checkpoint()

    expired_active = replace(checkpoint, last_observed_at=1)
    skipped_counter = replace(
        checkpoint, next_fencing_token=checkpoint.next_fencing_token + 1
    )
    with pytest.raises(ClaimInvariantError, match="expired active"):
        _restore(expired_active)
    with pytest.raises(ClaimInvariantError, match="skip fencing tokens"):
        _restore(skipped_counter)


def test_restore_rejects_out_of_order_claim_ids() -> None:
    ledger = ClaimLedger((0, 2), clock=LogicalClock())
    first = ledger.claim_next("first", 1)
    second = ledger.claim_next("second", 1)
    checkpoint = ledger.checkpoint()

    first_as_second = {
        "claim_id": second.claim_id,
        "owner": second.owner,
        "start": second.span.start,
        "end": second.span.end,
        "fencing_token": second.fencing_token,
        "acquired_at": second.acquired_at,
    }
    second_as_first = {
        "claim_id": first.claim_id,
        "owner": first.owner,
        "start": first.span.start,
        "end": first.span.end,
        "fencing_token": first.fencing_token,
        "acquired_at": first.acquired_at,
    }
    corrupted = _replace_event(
        checkpoint,
        1,
        stream="claim:2",
        payload_changes=first_as_second,
    )
    corrupted = _replace_event(
        corrupted,
        2,
        stream="claim:1",
        payload_changes=second_as_first,
    )

    with pytest.raises(ClaimInvariantError, match="claim IDs"):
        _restore(corrupted)


def test_restore_rejects_overlapping_claims_after_coherent_event_validation() -> None:
    ledger = ClaimLedger((0, 2), clock=LogicalClock())
    first = ledger.claim_next("first", 1)
    ledger.claim_next("second", 1)
    checkpoint = ledger.checkpoint()
    overlapping_second = replace(checkpoint.claims[1], span=first.span)
    corrupted = replace(
        checkpoint,
        claims=(checkpoint.claims[0], overlapping_second),
    )
    corrupted = _replace_event(
        corrupted,
        2,
        payload_changes={"start": first.span.start, "end": first.span.end},
    )

    with pytest.raises(ClaimInvariantError, match="overlap"):
        _restore(corrupted)
