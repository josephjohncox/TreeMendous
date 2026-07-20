"""Contracts for the private deterministic work-claim ledger."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest

from treemendous.applications._shared.claiming import (
    ClaimInvariantError,
    ClaimLedger,
    ClaimRequestConflictError,
    ClaimState,
    ClaimUnavailableError,
    ExpiredClaimError,
    ForeignClaimError,
    InvalidClaimError,
    StaleClaimError,
    TerminalClaimError,
    WorkClaim,
)
from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications._shared.events import EventLog, freeze_metadata
from treemendous.domain import ManagedDomain, Span


def _corrupt_claim_event(
    checkpoint: Any,
    sequence: int,
    *,
    payload_changes: dict[str, Any] | None = None,
    **event_changes: Any,
) -> Any:
    events = list(checkpoint.events.events)
    event = events[sequence - 1]
    if payload_changes is not None:
        payload = dict(event.payload)
        payload.update(payload_changes)
        event_changes["payload"] = tuple(payload.items())
    altered_event = replace(event, **event_changes)
    events[sequence - 1] = altered_event

    requests = list(checkpoint.events.requests)
    request_index = next(
        index
        for index, request in enumerate(requests)
        if request.event_sequence == sequence
    )
    requests[request_index] = replace(
        requests[request_index],
        key=altered_event.idempotency_key,
        kind=altered_event.kind,
        payload=altered_event.payload,
        occurred_at=altered_event.occurred_at,
    )
    event_checkpoint = replace(
        checkpoint.events,
        events=tuple(events),
        requests=tuple(requests),
    )
    return replace(checkpoint, events=event_checkpoint)


def test_claims_are_earliest_and_completion_records_only_metadata() -> None:
    clock = LogicalClock(10)
    ledger = ClaimLedger((0, 12), clock=clock)

    first = ledger.claim_next("worker-a", 4)
    second = ledger.claim_next("worker-b", 3)
    completed = ledger.complete(first, result={"matches": [7, 9]})

    assert first.span == Span(0, 4)
    assert second.span == Span(4, 7)
    assert first.claim_id == 1
    assert second.claim_id == 2
    assert first.fencing_token == 1
    assert second.fencing_token == 2
    assert completed.state is ClaimState.COMPLETED
    expected_result = freeze_metadata({"matches": [7, 9]})
    expected_available = (Span(7, 12),)
    assert completed.result == expected_result
    assert ledger.snapshot().available == expected_available


def test_abandon_returns_work_but_never_reuses_ids_or_tokens() -> None:
    ledger = ClaimLedger((0, 4), clock=LogicalClock())
    original = ledger.claim_next("worker", 4)
    abandoned = ledger.abandon(original)
    replacement = ledger.claim_next("other", 4)

    assert abandoned.state is ClaimState.ABANDONED
    assert replacement.span == original.span
    assert replacement.claim_id == 2
    assert replacement.fencing_token == 2
    with pytest.raises(TerminalClaimError):
        ledger.abandon(abandoned)


def test_expiry_reclaims_work_and_renewal_fences_old_revision() -> None:
    clock = LogicalClock(5)
    ledger = ClaimLedger((0, 3), clock=clock)
    original = ledger.claim_next("worker", 3, ttl=3)

    clock.advance()
    renewed = ledger.renew(original, ttl=4)
    assert renewed.revision == 2
    assert renewed.fencing_token == 2
    assert renewed.expires_at == 10
    with pytest.raises(StaleClaimError, match="old revision"):
        ledger.complete(original)

    clock.advance(4)
    expired = ledger.expire()
    assert len(expired) == 1
    assert expired[0].state is ClaimState.EXPIRED
    assert ledger.claim_next("replacement", 3).span == Span(0, 3)
    with pytest.raises(ExpiredClaimError):
        ledger.renew(renewed, ttl=1)


def test_request_id_is_idempotent_and_conflicting_reuse_is_atomic() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 8), clock=clock)
    original = ledger.claim_next(
        "worker", 2, ttl=5, not_before=1, request_id="request-1"
    )

    assert (
        ledger.claim_next("worker", 2, ttl=5, not_before=1, request_id="request-1")
        == original
    )
    before_claims = ledger._claims.copy()
    before_available = ledger._free.snapshot()
    before_events = ledger.events()
    clock.advance(5)
    with pytest.raises(ClaimRequestConflictError):
        ledger.claim_next("worker", 3, ttl=5, not_before=1, request_id="request-1")

    assert ledger._claims == before_claims
    assert ledger._free.snapshot() == before_available
    assert ledger.events() == before_events
    assert (
        ledger.claim_next("worker", 2, ttl=5, not_before=1, request_id="request-1")
        == original
    )


def test_foreign_owner_ledger_and_terminal_handles_are_rejected() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 4), clock=clock)
    other = ClaimLedger((0, 4), clock=clock)
    claim = ledger.claim_next("owner", 2)

    with pytest.raises(ForeignClaimError, match="owner"):
        ledger.complete(claim, owner="intruder")
    with pytest.raises(ForeignClaimError, match="another ledger"):
        other.complete(claim)
    completed = ledger.complete(claim)
    with pytest.raises(TerminalClaimError):
        ledger.complete(completed)


def test_rejected_handles_do_not_expire_unrelated_claims() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 4), clock=clock)
    ledger.claim_next("expiring", 1, ttl=1)
    old_handle = ledger.claim_next("owner", 1)
    current = ledger.renew(old_handle, ttl=10)
    before_claims = ledger._claims.copy()
    before_available = ledger._free.snapshot()
    before_events = ledger.events()
    clock.advance()

    with pytest.raises(InvalidClaimError, match="unknown"):
        ledger.complete(replace(current, claim_id=99))
    with pytest.raises(ForeignClaimError, match="owner"):
        ledger.complete(current, owner="intruder")
    with pytest.raises(StaleClaimError, match="old revision"):
        ledger.complete(old_handle)

    assert ledger._claims == before_claims
    assert ledger._free.snapshot() == before_available
    assert ledger.events() == before_events


def test_claim_value_and_transition_validation_is_atomic() -> None:
    base = WorkClaim("ledger", 1, "owner", Span(0, 1), 1, 0, None)
    invalid_values = (
        lambda: replace(base, owner=""),
        lambda: replace(base, claim_id=0),
        lambda: replace(base, expires_at=0),
        lambda: replace(base, state="active"),  # type: ignore[arg-type]
        lambda: replace(base, request_id=""),
        lambda: replace(base, result=[]),  # type: ignore[arg-type]
        lambda: replace(base, result=(("bad",),)),  # type: ignore[arg-type]
        lambda: replace(base, result=((1, "bad"),)),  # type: ignore[arg-type]
        lambda: replace(base, result=(("a", 1), ("a", 2))),
        lambda: replace(base, result=(("value", 1),)),
    )
    for invalid_value in invalid_values:
        with pytest.raises((TypeError, ValueError)):
            invalid_value()

    with pytest.raises(TypeError, match="clock"):
        ClaimLedger((0, 2), clock=object())  # type: ignore[arg-type]
    ledger = ClaimLedger((0, 2), clock=LogicalClock())
    claim = ledger.claim_next("owner", 1)
    before = ledger.snapshot()
    with pytest.raises(TypeError, match="WorkClaim"):
        ledger.complete(object())  # type: ignore[arg-type]
    with pytest.raises(InvalidClaimError, match="unknown"):
        ledger.complete(replace(claim, claim_id=99))
    with pytest.raises(ValueError, match="owner"):
        ledger.complete(claim, owner="")
    with pytest.raises(ValueError, match="greater than zero"):
        ledger.renew(claim, ttl=0)
    with pytest.raises(TypeError, match="payload values"):
        ledger.complete(claim, result={"bad": object()})
    assert ledger.snapshot() == before


def test_expiration_and_lifecycle_event_failures_are_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expiry_clock = LogicalClock()
    expiry_ledger = ClaimLedger((0, 2), clock=expiry_clock)
    expiry_ledger.claim_next("expiring", 1, ttl=1)
    lifecycle_ledger = ClaimLedger((0, 2), clock=LogicalClock())
    lifecycle_claim = lifecycle_ledger.claim_next("owner", 1)

    expiry_before = (
        expiry_ledger._claims.copy(),
        expiry_ledger._free.snapshot(),
        expiry_ledger.events(),
    )
    lifecycle_before = (
        lifecycle_ledger._claims.copy(),
        lifecycle_ledger._free.snapshot(),
        lifecycle_ledger.events(),
    )
    expiry_clock.advance()

    def fail_append(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected append failure")

    monkeypatch.setattr(EventLog, "append", fail_append)
    with pytest.raises(RuntimeError, match="injected"):
        expiry_ledger.expire()
    with pytest.raises(RuntimeError, match="injected"):
        lifecycle_ledger.abandon(lifecycle_claim)

    assert expiry_ledger._claims == expiry_before[0]
    assert expiry_ledger._free.snapshot() == expiry_before[1]
    assert expiry_ledger.events() == expiry_before[2]
    assert lifecycle_ledger._claims == lifecycle_before[0]
    assert lifecycle_ledger._free.snapshot() == lifecycle_before[1]
    assert lifecycle_ledger.events() == lifecycle_before[2]


def test_checkpoint_restores_retry_counters_events_and_availability() -> None:
    clock = LogicalClock(2)
    ledger = ClaimLedger((Span(0, 4), Span(10, 14)), clock=clock)
    first = ledger.claim_next("a", 2, ttl=10, request_id="a-1")
    renewed = ledger.renew(first, ttl=12)
    second = ledger.claim_next("b", 2)
    ledger.abandon(second)
    checkpoint = ledger.checkpoint()

    restored = ClaimLedger.from_checkpoint(checkpoint, clock=clock)
    assert restored.snapshot() == ledger.snapshot()
    retry = restored.claim_next("a", 2, ttl=10, request_id="a-1")
    assert retry == renewed
    next_claim = restored.claim_next("c", 2)
    assert next_claim.claim_id == 3
    assert next_claim.fencing_token == 4
    assert len(restored.events().events) == 5


def test_checkpoint_validation_rejects_overlap_and_counter_reuse() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 6), clock=clock)
    ledger.claim_next("a", 2, ttl=3, request_id="a-1")
    ledger.claim_next("b", 2)
    checkpoint = ledger.checkpoint()

    overlap = replace(checkpoint.claims[1], span=checkpoint.claims[0].span)
    with pytest.raises(ClaimInvariantError):
        ClaimLedger.from_checkpoint(
            replace(checkpoint, claims=(checkpoint.claims[0], overlap)),
            clock=clock,
        )
    with pytest.raises(ClaimInvariantError, match="reuse"):
        ClaimLedger.from_checkpoint(
            replace(checkpoint, next_claim_id=2),
            clock=clock,
        )
    with pytest.raises(ClaimInvariantError, match="incomplete"):
        ClaimLedger.from_checkpoint(replace(checkpoint, requests=()), clock=clock)
    altered_request = replace(checkpoint.requests[0], ttl=4)
    with pytest.raises(ClaimInvariantError, match="TTL"):
        ClaimLedger.from_checkpoint(
            replace(checkpoint, requests=(altered_request,)),
            clock=clock,
        )


def test_checkpoint_rejects_duplicate_restored_fencing_tokens() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 1), clock=clock)
    original = ledger.claim_next("first", 1)
    ledger.abandon(original)
    ledger.claim_next("replacement", 1)
    checkpoint = ledger.checkpoint()

    duplicate = replace(
        checkpoint.claims[1],
        fencing_token=checkpoint.claims[0].fencing_token,
    )
    corrupted = replace(
        checkpoint,
        claims=(checkpoint.claims[0], duplicate),
    )
    corrupted = _corrupt_claim_event(
        corrupted,
        3,
        payload_changes={"fencing_token": original.fencing_token},
    )

    with pytest.raises(ClaimInvariantError, match="fencing tokens"):
        ClaimLedger.from_checkpoint(corrupted, clock=clock)


def test_checkpoint_rejects_future_transition_idempotency_key() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 1), clock=clock)
    claim = ledger.claim_next("owner", 1, ttl=1)
    ledger.renew(claim, ttl=2)
    checkpoint = ledger.checkpoint()
    corrupted = _corrupt_claim_event(
        checkpoint,
        1,
        idempotency_key="3:expired",
    )

    with pytest.raises(ClaimInvariantError, match="idempotency key"):
        ClaimLedger.from_checkpoint(corrupted, clock=clock)

    requests = list(checkpoint.events.requests)
    requests[0] = replace(requests[0], expected_version=None)
    missing_expected_version = replace(
        checkpoint,
        events=replace(checkpoint.events, requests=tuple(requests)),
    )
    with pytest.raises(ClaimInvariantError, match="idempotency request"):
        ClaimLedger.from_checkpoint(missing_expected_version, clock=clock)


def test_checkpoint_rejects_terminal_transition_after_expiry() -> None:
    clock = LogicalClock()
    ledger = ClaimLedger((0, 1), clock=clock)
    claim = ledger.claim_next("owner", 1, ttl=1)
    ledger.complete(claim)
    checkpoint = ledger.checkpoint()
    corrupted = _corrupt_claim_event(checkpoint, 2, occurred_at=2)
    corrupted = replace(corrupted, last_observed_at=2)

    with pytest.raises(ClaimInvariantError, match="after expiry"):
        ClaimLedger.from_checkpoint(corrupted, clock=LogicalClock(2))


def test_checkpoint_rejects_nonmonotonic_global_event_timestamps() -> None:
    clock = LogicalClock(2)
    ledger = ClaimLedger((0, 2), clock=clock)
    ledger.claim_next("first", 1)
    clock.advance()
    ledger.claim_next("second", 1)
    checkpoint = ledger.checkpoint()
    corrupted = _corrupt_claim_event(checkpoint, 2, occurred_at=1)

    with pytest.raises(ClaimInvariantError, match="globally monotonic"):
        ClaimLedger.from_checkpoint(corrupted, clock=clock)


def test_checkpoint_rejects_foreign_types_clock_and_request_corruption() -> None:
    clock = LogicalClock(5)
    ledger = ClaimLedger((0, 4), clock=clock)
    ledger.claim_next("owner", 2, ttl=3, request_id="r1")
    checkpoint = ledger.checkpoint()

    with pytest.raises(TypeError, match="ClaimLedgerCheckpoint"):
        ClaimLedger.from_checkpoint(object(), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ManagedDomain"):
        ClaimLedger.from_checkpoint(
            replace(checkpoint, domain=(0, 4)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ClaimInvariantError, match="positive"):
        ClaimLedger.from_checkpoint(
            replace(checkpoint, next_fencing_token=0), clock=clock
        )
    with pytest.raises(ClaimInvariantError, match="predates"):
        ClaimLedger.from_checkpoint(checkpoint, clock=LogicalClock(4))
    foreign = replace(checkpoint.claims[0], ledger_id="other")
    with pytest.raises(ClaimInvariantError, match="another ledger"):
        ClaimLedger.from_checkpoint(replace(checkpoint, claims=(foreign,)), clock=clock)
    with pytest.raises(ClaimInvariantError, match="duplicate claim"):
        ClaimLedger.from_checkpoint(
            replace(
                checkpoint,
                claims=(checkpoint.claims[0], checkpoint.claims[0]),
            ),
            clock=clock,
        )
    outside = replace(checkpoint.claims[0], span=Span(10, 12))
    with pytest.raises(ClaimInvariantError, match="outside"):
        ClaimLedger.from_checkpoint(replace(checkpoint, claims=(outside,)), clock=clock)
    with pytest.raises(TypeError, match="requests"):
        ClaimLedger.from_checkpoint(
            replace(checkpoint, requests=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    request = checkpoint.requests[0]
    with pytest.raises(ClaimInvariantError, match="duplicate claim request"):
        ClaimLedger.from_checkpoint(
            replace(checkpoint, requests=(request, request)), clock=clock
        )
    wrong_claim = replace(request, claim_id=99)
    with pytest.raises(ClaimInvariantError, match="does not match"):
        ClaimLedger.from_checkpoint(
            replace(checkpoint, requests=(wrong_claim,)), clock=clock
        )
    wrong_owner = replace(request, owner="other")
    with pytest.raises(ClaimInvariantError, match="arguments"):
        ClaimLedger.from_checkpoint(
            replace(checkpoint, requests=(wrong_owner,)), clock=clock
        )


def test_diagnostics_and_invariants_account_for_disjoint_domain() -> None:
    ledger = ClaimLedger(
        ManagedDomain((Span(0, 3), Span(10, 15))), clock=LogicalClock()
    )
    active = ledger.claim_next("a", 2)
    completed = ledger.claim_next("b", 1)
    ledger.complete(completed)

    snapshot = ledger.snapshot()
    expected_available = (Span(10, 15),)
    assert snapshot.available == expected_available
    assert snapshot.diagnostics.total_work == 8
    assert snapshot.diagnostics.available_work == 5
    assert snapshot.diagnostics.active_claims == 1
    assert snapshot.diagnostics.completed_claims == 1
    assert not ledger.invariant_violations()
    ledger.validate()
    assert active.state is ClaimState.ACTIVE


def test_concurrent_claims_are_disjoint_and_exhaust_capacity() -> None:
    ledger = ClaimLedger((0, 40), clock=LogicalClock())

    def claim(index: int):
        try:
            return ledger.claim_next(f"worker-{index}", 1)
        except ClaimUnavailableError:
            return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(claim, range(80)))
    claims = [claim for claim in results if claim is not None]

    assert len(claims) == 40
    assert sorted(claim.span.start for claim in claims) == list(range(40))
    assert len({claim.claim_id for claim in claims}) == 40
    assert len({claim.fencing_token for claim in claims}) == 40
    with pytest.raises(ClaimUnavailableError):
        ledger.claim_next("late", 1)


def test_validation_and_clock_failures_do_not_mutate_state() -> None:
    clock = LogicalClock(10)
    ledger = ClaimLedger((0, 5), clock=clock)
    before = ledger.snapshot()

    with pytest.raises(ValueError, match="ttl"):
        ledger.claim_next("worker", 1, ttl=0)
    with pytest.raises(ValueError, match="not_after"):
        ledger.claim_next("worker", 1, not_before=3, not_after=3)
    with pytest.raises(ClaimUnavailableError):
        ledger.claim_next("worker", 6)
    assert ledger.snapshot().claims == before.claims
    assert ledger.snapshot().available == before.available

    # A custom backward clock is rejected before any transition.
    clock._value = 9
    with pytest.raises(ValueError, match="backwards"):
        ledger.claim_next("worker", 1)
