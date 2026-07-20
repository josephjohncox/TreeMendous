"""Public edge and adversarial contracts for numeric leasing applications."""

from __future__ import annotations

import ipaddress
from dataclasses import replace
from typing import cast

import pytest

from treemendous.applications._shared.clock import LogicalClock
from treemendous.applications._shared.leasing import (
    ExpiredLeaseError,
    ForeignLeaseError,
    InvalidLeaseError,
    Lease,
    LeasePool,
    LeasePoolCheckpoint,
    LeaseRequestConflictError,
    LeaseState,
    StaleLeaseError,
)
from treemendous.applications.leasing._common import (
    GroupCheckpoint,
    NumericLease,
    PoolGroup,
    ProcessClock,
    ScopedPoolCheckpoint,
)
from treemendous.applications.leasing.database_ids import (
    CommittedIdBatch,
    CommittedIdError,
    DatabaseIdPool,
    DatabaseIdUnavailableError,
)
from treemendous.applications.leasing.game_regions import (
    GameRegionPool,
    RegionAdjacencyError,
    RegionHandoffError,
    RegionUnavailableError,
)
from treemendous.applications.leasing.numeric_ip_pools import (
    AddressPoolCheckpoint,
    AddressUnavailableError,
    NumericIPAddressPool,
)
from treemendous.applications.leasing.software_seats import (
    EntitlementError,
    SeatUnavailableError,
    SoftwareSeatPool,
    UnknownProductError,
)
from treemendous.applications.leasing.vlan_tags import (
    VlanPoolCheckpoint,
    VlanScopeError,
    VlanTagPool,
    VlanUnavailableError,
)
from treemendous.applications.leasing.warehouse_bins import (
    BinCompatibilityError,
    BinRequestConflictError,
    BinUnavailableError,
    BinZone,
    WarehouseBinPool,
)
from treemendous.domain import ManagedDomain, Span


def test_shared_lease_value_and_pool_input_validators() -> None:
    valid = {
        "pool_id": "pool",
        "owner": "owner",
        "resource": Span(0, 1),
        "fencing_token": 1,
        "acquired_at": 0,
        "expires_at": 1,
    }
    invalid = (
        ({"pool_id": 1}, TypeError, "pool_id"),
        ({"owner": ""}, ValueError, "owner"),
        ({"fencing_token": 0}, ValueError, "fencing_token"),
        ({"expires_at": 0}, ValueError, "expires_at"),
        ({"state": "active"}, TypeError, "LeaseState"),
        ({"revision": 0}, ValueError, "revision"),
        ({"request_id": ""}, ValueError, "request_id"),
    )
    for changes, error, message in invalid:
        with pytest.raises(error, match=message):
            Lease(**(valid | changes))  # type: ignore[arg-type]

    active = Lease("pool", "owner", Span(0, 1), 1, 0, 1)
    assert active.span == active.resource
    assert active.active
    assert not replace(active, state=LeaseState.RELEASED).active

    with pytest.raises(TypeError, match="clock"):
        LeasePool((0, 1), clock=object())  # type: ignore[arg-type]
    pool = LeasePool(ManagedDomain((Span(0, 3),)), clock=LogicalClock())
    assert list(pool.allowed_spans) == [Span(0, 3)]

    invalid_acquires = (
        ((1,), {"ttl": 1}, TypeError, "owner"),
        (("",), {"ttl": 1}, ValueError, "owner"),
        (("owner",), {"ttl": 1, "exact_span": [0, 1]}, TypeError, "exact_span"),
        (
            ("owner",),
            {"ttl": 1, "exact_span": (0, "x")},
            TypeError,
            "invalid exact_span",
        ),
        (("owner",), {"ttl": 1, "request_id": ""}, ValueError, "request_id"),
    )
    for args, kwargs, error, message in invalid_acquires:
        with pytest.raises(error, match=message):
            pool.acquire(*args, **kwargs)

    exact = pool.acquire("owner", ttl=2, exact_span=Span(0, 2))
    assert exact.resource == Span(0, 2)
    with pytest.raises(TypeError, match="handle"):
        pool.release(object())  # type: ignore[arg-type]
    unknown = replace(exact, fencing_token=99)
    with pytest.raises(InvalidLeaseError, match="not issued"):
        pool.release(unknown)
    wrong_owner = replace(exact, owner="other")
    with pytest.raises(ForeignLeaseError, match="owner"):
        pool.release(wrong_owner)


def test_shared_checkpoint_rejects_inconsistent_public_evidence() -> None:
    clock = LogicalClock(2)
    pool = LeasePool((0, 4), clock=clock)
    lease = pool.acquire(
        "owner", ttl=5, size=2, exact_span=(0, 2), request_id="request"
    )
    checkpoint = pool.checkpoint()
    request = checkpoint.requests[0]

    with pytest.raises(ValueError, match="allowed_spans"):
        LeasePoolCheckpoint("pool", (), (), (), 1, 0)
    with pytest.raises(TypeError, match="checkpoint"):
        LeasePool.from_checkpoint(object(), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="clock"):
        LeasePool.from_checkpoint(checkpoint, clock=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="normalized"):
        LeasePool.from_checkpoint(
            replace(
                checkpoint,
                allowed_spans=(Span(0, 2), Span(2, 4)),
            ),
            clock=clock,
        )
    with pytest.raises(TypeError, match="Lease values"):
        LeasePool.from_checkpoint(
            replace(checkpoint, leases=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="another pool"):
        LeasePool.from_checkpoint(
            replace(checkpoint, leases=(replace(lease, pool_id="foreign"),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="outside"):
        LeasePool.from_checkpoint(
            replace(checkpoint, leases=(replace(lease, resource=Span(3, 5)),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="tokens must be unique"):
        LeasePool.from_checkpoint(
            replace(checkpoint, leases=(lease, lease)), clock=clock
        )

    with pytest.raises(TypeError, match="invalid record"):
        LeasePool.from_checkpoint(
            replace(checkpoint, requests=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="request IDs must be unique"):
        LeasePool.from_checkpoint(
            replace(checkpoint, requests=(request, request)), clock=clock
        )
    with pytest.raises(ValueError, match="identify its lease"):
        LeasePool.from_checkpoint(
            replace(
                checkpoint,
                requests=(replace(request, lease_token=99),),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="conflicts with its lease"):
        LeasePool.from_checkpoint(
            replace(checkpoint, requests=(replace(request, owner="other"),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="exact request"):
        LeasePool.from_checkpoint(
            replace(
                checkpoint,
                requests=(replace(request, exact_span=Span(1, 3)),),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="missing its request"):
        LeasePool.from_checkpoint(replace(checkpoint, requests=()), clock=clock)

    overlapping = replace(
        checkpoint,
        leases=(lease, replace(lease, fencing_token=2, resource=Span(1, 3))),
        requests=(),
        next_fencing_token=3,
    )
    overlapping = replace(
        overlapping,
        leases=tuple(replace(item, request_id=None) for item in overlapping.leases),
    )
    with pytest.raises(ValueError, match="disjoint"):
        LeasePool.from_checkpoint(overlapping, clock=clock)


def test_shared_lifecycle_rejects_changed_identity_and_terminal_states() -> None:
    clock = LogicalClock()
    pool = LeasePool((0, 2), clock=clock)
    lease = pool.acquire("owner", ttl=2, request_id="stable")

    with pytest.raises(LeaseRequestConflictError):
        pool.acquire("changed", ttl=2, request_id="stable")
    with pytest.raises(ValueError, match="ttl"):
        pool.renew(lease, ttl=0)
    with pytest.raises(ValueError, match="new_owner"):
        pool.transfer(lease, "", ttl=1)

    renewed = pool.renew(lease, ttl=2)
    with pytest.raises(StaleLeaseError, match="old"):
        pool.transfer(lease, "next", ttl=2)
    released = pool.release(renewed)
    with pytest.raises(StaleLeaseError, match="released"):
        pool.transfer(released, "next", ttl=2)

    expiring = pool.acquire("expiring", ttl=1)
    clock.advance()
    with pytest.raises(ExpiredLeaseError):
        pool.release(expiring)


def test_common_helpers_group_validation_and_forged_fence_evidence() -> None:
    assert ProcessClock().now() >= 0
    clock = LogicalClock()
    with pytest.raises(ValueError, match="at least one"):
        PoolGroup({}, clock=clock)

    group = PoolGroup({"scope": (Span(0, 3),)}, clock=clock)
    handle = group.acquire("scope", "owner", ttl=3)
    assert handle.owner == "owner"
    assert handle.expires_at == 3
    assert handle.revision == 1

    with pytest.raises(ValueError, match="unknown"):
        group.pool("missing")
    with pytest.raises(TypeError, match="NumericLease"):
        group.renew(object(), ttl=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NumericLease"):
        group.release(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NumericLease"):
        group.transfer(object(), "next", ttl=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NumericLease"):
        group.validate_fence("key", object())  # type: ignore[arg-type]

    foreign_group = PoolGroup({"scope": (Span(0, 3),)}, clock=clock)
    foreign = foreign_group.acquire("scope", "owner", ttl=3)
    with pytest.raises(ForeignLeaseError, match="lineage"):
        group.validate_fence("key", foreign)

    unknown = NumericLease("scope", replace(handle.lease, fencing_token=99))
    with pytest.raises(InvalidLeaseError, match="not issued"):
        group.validate_fence("key", unknown)
    altered = NumericLease("scope", replace(handle.lease, owner="forged"))
    with pytest.raises(InvalidLeaseError, match="differs"):
        group.validate_fence("key", altered)

    with pytest.raises(ValueError, match="scopes"):
        group.require_domains({"other": (Span(0, 3),)})
    with pytest.raises(ValueError, match="domain"):
        group.require_domains({"scope": (Span(0, 2),)})


def test_common_checkpoint_and_wrapper_validators() -> None:
    clock = LogicalClock()
    group = PoolGroup({"scope": (Span(0, 3),)}, clock=clock)
    checkpoint = group.checkpoint()
    scoped = checkpoint.pools[0]

    with pytest.raises(ValueError, match="scope"):
        NumericLease("", group.acquire("scope", "owner", ttl=1).lease)
    with pytest.raises(TypeError, match="Lease"):
        NumericLease("scope", object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pool"):
        ScopedPoolCheckpoint("scope", "lineage", (Span(0, 1),), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="normalized"):
        ScopedPoolCheckpoint(
            "scope",
            scoped.source_pool_id,
            (Span(0, 1), Span(1, 3)),
            replace(scoped.pool, allowed_spans=(Span(0, 1), Span(1, 3))),
        )
    with pytest.raises(ValueError, match="domain does not match"):
        ScopedPoolCheckpoint("scope", scoped.source_pool_id, (Span(0, 2),), scoped.pool)
    with pytest.raises(ValueError, match="source_pool_id"):
        ScopedPoolCheckpoint("scope", "", scoped.allowed_spans, scoped.pool)
    with pytest.raises(ValueError, match="lineage"):
        ScopedPoolCheckpoint("scope", "other", scoped.allowed_spans, scoped.pool)

    with pytest.raises(TypeError, match="GroupCheckpoint"):
        PoolGroup.restore(object(), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        PoolGroup.restore(GroupCheckpoint(()), clock=clock)
    with pytest.raises(TypeError, match="ScopedPoolCheckpoint"):
        PoolGroup.restore(GroupCheckpoint((object(),)), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        PoolGroup.restore(GroupCheckpoint((scoped, scoped)), clock=clock)


def test_region_public_validators_unavailability_and_adjacency_states() -> None:
    clock = LogicalClock()
    with pytest.raises(ValueError, match="at least one shard"):
        GameRegionPool({}, clock=clock)
    with pytest.raises(TypeError, match="bounds"):
        GameRegionPool({"bad": [1, 2]}, clock=clock)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="must not exceed"):
        GameRegionPool({"bad": (2, 1)}, clock=clock)
    with pytest.raises(ValueError, match="shard"):
        GameRegionPool({"": (1, 2)}, clock=clock)

    engine = GameRegionPool({"west": (1, 2), "east": (1, 2)}, clock=clock)
    invalid_calls = (
        ({"shard": "missing", "owner": "owner", "ttl": 1}, ValueError, "unknown"),
        ({"shard": "west", "owner": "", "ttl": 1}, ValueError, "owner"),
        ({"shard": "west", "owner": "owner", "ttl": 0}, ValueError, "ttl"),
        (
            {"shard": "west", "owner": "owner", "ttl": 1, "count": 0},
            ValueError,
            "count",
        ),
        (
            {"shard": "west", "owner": "owner", "ttl": 1, "request_id": ""},
            ValueError,
            "request_id",
        ),
        (
            {
                "shard": "west",
                "owner": "owner",
                "ttl": 1,
                "adjacent_to": object(),
            },
            TypeError,
            "RegionLease",
        ),
    )
    for kwargs, error, message in invalid_calls:
        with pytest.raises(error, match=message):
            engine.acquire(**kwargs)  # type: ignore[arg-type]

    first = engine.acquire("west", "owner", ttl=5, start_region=1, request_id="exact")
    assert (
        engine.acquire("west", "owner", ttl=5, start_region=1, request_id="exact")
        == first
    )
    with pytest.raises(LeaseRequestConflictError):
        engine.acquire("west", "changed", ttl=5, start_region=1, request_id="exact")
    with pytest.raises(ValueError, match="mutually exclusive"):
        engine.acquire(
            "west",
            "owner",
            ttl=1,
            start_region=2,
            adjacent_to=first,
        )

    second = engine.acquire("west", "owner", ttl=5, adjacent_to=first)
    with pytest.raises(RegionAdjacencyError, match="no available"):
        engine.acquire("west", "owner", ttl=5, adjacent_to=first)
    with pytest.raises(RegionAdjacencyError, match="same shard and owner"):
        engine.acquire("east", "owner", ttl=5, adjacent_to=first)
    with pytest.raises(RegionUnavailableError):
        engine.acquire("west", "other", ttl=1)
    with pytest.raises(RegionUnavailableError):
        engine.acquire("east", "other", ttl=1, start_region=3)
    with pytest.raises(ValueError, match="outside"):
        engine.validate_fence(second, 3)

    engine.release(first)
    with pytest.raises(RegionAdjacencyError, match="not active"):
        engine.acquire("west", "owner", ttl=1, adjacent_to=first)


def test_region_handoff_retries_require_unchanged_identity() -> None:
    clock = LogicalClock()
    engine = GameRegionPool({"west": (1, 3)}, clock=clock)
    source = engine.acquire("west", "source", ttl=5)

    with pytest.raises(TypeError, match="RegionLease"):
        engine.handoff(
            cast(NumericLease, object()),
            "target",
            ttl=2,
            request_id="bad",
        )
    transferred = engine.handoff(source, "target", ttl=2, request_id="handoff")
    assert transferred.resource == source.resource
    assert engine.handoff(source, "target", ttl=2, request_id="handoff") == transferred
    with pytest.raises(RegionHandoffError, match="different arguments"):
        engine.handoff(source, "other", ttl=2, request_id="handoff")
    with pytest.raises(RegionHandoffError, match="different arguments"):
        engine.handoff(source, "target", ttl=3, request_id="handoff")
    with pytest.raises(ValueError, match="new_owner"):
        engine.handoff(source, "", ttl=2, request_id="other")
    with pytest.raises(ValueError, match="request_id"):
        engine.handoff(source, "target", ttl=2, request_id="")


def test_region_restore_rejects_shard_request_and_handoff_corruption() -> None:
    clock = LogicalClock()
    engine = GameRegionPool({"west": (1, 8)}, clock=clock)
    anchor = engine.acquire("west", "owner", ttl=10, start_region=1, request_id="exact")
    engine.acquire("west", "owner", ttl=10, adjacent_to=anchor, request_id="adjacent")
    engine.acquire("west", "owner", ttl=10, request_id="automatic")
    engine.handoff(anchor, "next", ttl=10, request_id="handoff")
    checkpoint = engine.checkpoint()
    requests = {item.request_id: item for item in checkpoint.requests}
    adjacent = requests["adjacent"]
    exact = requests["exact"]
    handoff = checkpoint.handoffs[0]

    with pytest.raises(TypeError, match="RegionPoolCheckpoint"):
        GameRegionPool.from_checkpoint(object(), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, shards=checkpoint.shards * 2), clock=clock
        )
    with pytest.raises(TypeError, match="bounds"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, shards=(("west", [1, 8]),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="at least one shard"):
        GameRegionPool.from_checkpoint(replace(checkpoint, shards=()), clock=clock)
    with pytest.raises(TypeError, match="invalid request"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, requests=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="request IDs must be unique"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, requests=(exact, exact)), clock=clock
        )
    with pytest.raises(ValueError, match="unknown shard"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, requests=(replace(exact, shard="missing"),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="invalid selection"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, requests=(replace(exact, selection_mode="bad"),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="start conflicts"):
        GameRegionPool.from_checkpoint(
            replace(
                checkpoint,
                requests=(replace(exact, selection_mode="automatic"),),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="start conflicts"):
        GameRegionPool.from_checkpoint(
            replace(
                checkpoint,
                requests=(replace(exact, result_resource=Span(2, 3)),),
            ),
            clock=clock,
        )
    with pytest.raises(TypeError, match="invalid anchor"):
        GameRegionPool.from_checkpoint(
            replace(
                checkpoint,
                requests=(replace(adjacent, anchor=object()),),  # type: ignore[arg-type]
            ),
            clock=clock,
        )
    assert adjacent.anchor is not None
    with pytest.raises(ValueError, match="anchor conflicts"):
        GameRegionPool.from_checkpoint(
            replace(
                checkpoint,
                requests=(
                    replace(
                        adjacent,
                        anchor=replace(adjacent.anchor, owner="different"),
                    ),
                ),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="anchor conflicts"):
        GameRegionPool.from_checkpoint(
            replace(
                checkpoint,
                requests=(
                    replace(
                        adjacent,
                        anchor=replace(
                            adjacent.anchor,
                            pool_id="foreign-lineage",
                        ),
                    ),
                ),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="count conflicts"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, requests=(replace(exact, count=2),)), clock=clock
        )
    with pytest.raises(ValueError, match="lease history"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, requests=(replace(exact, result_token=999),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="agree with lease history"):
        GameRegionPool.from_checkpoint(replace(checkpoint, requests=()), clock=clock)

    with pytest.raises(TypeError, match="invalid handoff"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, handoffs=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="handoff request IDs must be unique"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, handoffs=(handoff, handoff)), clock=clock
        )
    with pytest.raises(ValueError, match="handoff record"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, handoffs=(replace(handoff, result_token=999),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="handoff result identity"):
        GameRegionPool.from_checkpoint(
            replace(
                checkpoint,
                handoffs=(
                    replace(
                        handoff,
                        result_identity=replace(
                            handoff.result_identity,
                            pool_id="foreign-lineage",
                        ),
                    ),
                ),
            ),
            clock=clock,
        )


def test_region_restore_rejects_handoff_rebound_to_later_matching_lease() -> None:
    clock = LogicalClock()
    engine = GameRegionPool({"west": (1, 1)}, clock=clock)
    source = engine.acquire("west", "source", ttl=10)
    result = engine.handoff(source, "target", ttl=10, request_id="handoff")
    engine.release(result)
    later = engine.acquire("west", "target", ttl=10, start_region=1)
    checkpoint = engine.checkpoint()
    forged_handoff = replace(checkpoint.handoffs[0], result_token=later.token)

    with pytest.raises(ValueError, match="handoff result identity"):
        GameRegionPool.from_checkpoint(
            replace(checkpoint, handoffs=(forged_handoff,)),
            clock=clock,
        )


def test_database_public_lifecycle_reuse_exhaustion_and_identity() -> None:
    clock = LogicalClock()
    with pytest.raises(ValueError, match="namespace"):
        DatabaseIdPool("", clock=clock)
    with pytest.raises(ValueError, match="must not exceed"):
        DatabaseIdPool("bad", minimum_id=2, maximum_id=1, clock=clock)

    pool = DatabaseIdPool("ids", maximum_id=6, clock=clock)
    with pytest.raises(ValueError, match="count"):
        pool.acquire("owner", ttl=1, count=0)
    with pytest.raises(DatabaseIdUnavailableError, match="reusable"):
        pool.acquire("owner", ttl=1, reusable=True)

    first = pool.acquire("one", ttl=5, request_id="first")
    middle = pool.acquire("middle", ttl=5)
    tail = pool.acquire("tail", ttl=5, count=3)
    with pytest.raises(LeaseRequestConflictError):
        pool.acquire("changed", ttl=5, request_id="first")
    with pytest.raises(DatabaseIdUnavailableError, match="exhausted"):
        pool.acquire("overflow", ttl=1, count=2)

    pool.release(first)
    pool.release(tail)
    reused = pool.acquire("reuse", ttl=3, count=2, reusable=True)
    assert reused.resource == Span(3, 5)
    assert list(pool.snapshot().reusable_spans) == [Span(1, 2), Span(5, 6)]
    renewed = pool.renew(middle, ttl=8)
    assert renewed.revision == 2
    with pytest.raises(ValueError, match="outside"):
        pool.validate_fence(reused, 6)

    committed = pool.commit(renewed)
    assert committed.resource == renewed.resource
    assert committed.token == renewed.token
    assert pool.commit(renewed) == committed
    with pytest.raises(CommittedIdError, match="altered"):
        pool.commit(NumericLease("ids", replace(renewed.lease, owner="forged")))
    with pytest.raises(CommittedIdError, match="renewable"):
        pool.renew(renewed, ttl=1)


def test_database_restore_validates_bounds_commits_and_reusable_spans() -> None:
    clock = LogicalClock(5)
    pool = DatabaseIdPool("ids", maximum_id=10, clock=clock)
    committed = pool.commit(pool.acquire("permanent", ttl=5, request_id="committed"))
    active = pool.acquire("active", ttl=5, count=2, request_id="active")
    checkpoint = pool.checkpoint()

    with pytest.raises(TypeError, match="DatabaseIdCheckpoint"):
        DatabaseIdPool.from_checkpoint(object(), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not exceed"):
        DatabaseIdPool.from_checkpoint(replace(checkpoint, minimum_id=11), clock=clock)
    for next_id in (0, 12):
        with pytest.raises(ValueError, match="outside the ID domain"):
            DatabaseIdPool.from_checkpoint(
                replace(checkpoint, next_monotonic_id=next_id), clock=clock
            )

    with pytest.raises(TypeError, match="committed values"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, committed=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    invalid_source = CommittedIdBatch(
        "ids", cast(NumericLease, object()), committed.committed_at
    )
    with pytest.raises(TypeError, match="committed source"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, committed=(invalid_source,)), clock=clock
        )
    with pytest.raises(ValueError, match="namespace"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, committed=(replace(committed, namespace="other"),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="active revision"):
        DatabaseIdPool.from_checkpoint(
            replace(
                checkpoint,
                committed=(
                    replace(
                        committed,
                        source=NumericLease(
                            "ids",
                            replace(committed.source.lease, state=LeaseState.RELEASED),
                        ),
                    ),
                ),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="precede acquisition"):
        DatabaseIdPool.from_checkpoint(
            replace(
                checkpoint,
                committed=(replace(committed, committed_at=4),),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="absent"):
        DatabaseIdPool.from_checkpoint(
            replace(
                checkpoint,
                committed=(
                    replace(
                        committed,
                        source=NumericLease(
                            "ids",
                            replace(committed.source.lease, fencing_token=999),
                        ),
                    ),
                ),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="released pool history"):
        DatabaseIdPool.from_checkpoint(
            replace(
                checkpoint,
                committed=(
                    replace(
                        committed,
                        source=NumericLease(
                            "ids", replace(committed.source.lease, owner="changed")
                        ),
                    ),
                ),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="must not overlap"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, committed=(committed, committed)), clock=clock
        )

    with pytest.raises(TypeError, match="Span values"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, reusable_spans=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="normalized"):
        DatabaseIdPool.from_checkpoint(
            replace(
                checkpoint,
                reusable_spans=(Span(5, 6), Span(6, 7)),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="outside the ID domain"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, reusable_spans=(Span(11, 12),)), clock=clock
        )
    with pytest.raises(ValueError, match="active or committed"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, reusable_spans=(active.resource,)), clock=clock
        )


def test_database_restore_validates_idempotency_records() -> None:
    clock = LogicalClock()
    pool = DatabaseIdPool("ids", maximum_id=10, clock=clock)
    lease = pool.acquire("owner", ttl=5, count=2, request_id="request")
    checkpoint = pool.checkpoint()
    request = checkpoint.requests[0]

    with pytest.raises(TypeError, match="invalid record"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, requests=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(TypeError, match="reusable must be a bool"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, requests=(replace(request, reusable=1),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="outside the ID domain"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, requests=(replace(request, resource=Span(9, 12)),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="count conflicts"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, requests=(replace(request, count=1),)), clock=clock
        )
    with pytest.raises(ValueError, match="lease history"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, requests=(replace(request, owner="other"),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="request IDs must be unique"):
        DatabaseIdPool.from_checkpoint(
            replace(checkpoint, requests=(request, request)), clock=clock
        )
    with pytest.raises(ValueError, match="agree with lease history"):
        DatabaseIdPool.from_checkpoint(replace(checkpoint, requests=()), clock=clock)

    assert lease.resource == Span(1, 3)


def test_bin_zone_pool_and_acquisition_validators() -> None:
    clock = LogicalClock()
    with pytest.raises(ValueError, match="must not exceed"):
        BinZone(2, 1)
    with pytest.raises(ValueError, match="must not be empty"):
        BinZone(1, 2, size_classes=frozenset())
    with pytest.raises(ValueError, match="compatibility label"):
        BinZone(1, 2, hazards=frozenset({""}))
    with pytest.raises(TypeError, match="compatibility label"):
        BinZone(1, 2, size_classes=frozenset({1}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one bin zone"):
        WarehouseBinPool({}, clock=clock)
    with pytest.raises(TypeError, match="BinZone"):
        WarehouseBinPool({"A": object()}, clock=clock)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="zone"):
        WarehouseBinPool({"": BinZone(1, 2)}, clock=clock)

    zone = BinZone(
        1,
        2,
        size_classes=frozenset({"small", "large"}),
        hazards=frozenset({"general", "cold"}),
    )
    pool = WarehouseBinPool({"A": zone, "B": zone}, clock=clock)
    with pytest.raises(BinCompatibilityError):
        pool.acquire("missing", "owner", ttl=1)
    with pytest.raises(ValueError):
        pool.acquire("A", "owner", ttl=1, size_class="")
    with pytest.raises(ValueError):
        pool.acquire("A", "owner", ttl=1, hazard="")
    with pytest.raises(BinCompatibilityError):
        pool.acquire("A", "owner", ttl=1, size_class="bad")
    with pytest.raises(BinCompatibilityError):
        pool.acquire("A", "owner", ttl=1, hazard="bad")
    with pytest.raises(ValueError):
        pool.acquire("A", "owner", ttl=1, count=0)

    lease = pool.acquire(
        "A",
        "owner",
        ttl=4,
        size_class="large",
        hazard="cold",
        request_id="request",
    )
    assert lease.owner == "owner"
    assert lease.token == 1
    assert lease.expires_at == 4
    with pytest.raises(BinRequestConflictError):
        pool.acquire(
            "B",
            "owner",
            ttl=4,
            size_class="large",
            hazard="cold",
            request_id="request",
        )
    with pytest.raises(BinRequestConflictError):
        pool.acquire(
            "A",
            "owner",
            ttl=4,
            size_class="large",
            hazard="general",
            request_id="request",
        )
    with pytest.raises(LeaseRequestConflictError):
        pool.acquire(
            "A",
            "changed",
            ttl=4,
            size_class="large",
            hazard="cold",
            request_id="request",
        )
    with pytest.raises(BinUnavailableError):
        pool.acquire(
            "A",
            "outside",
            ttl=1,
            start_bin=3,
            size_class="large",
            hazard="cold",
        )
    with pytest.raises(ValueError, match="outside"):
        pool.validate_fence(lease, 2)
    renewed = pool.renew(lease, ttl=5)
    assert renewed.inner.revision == 2


def test_bin_restore_rejects_policy_metadata_and_history_corruption() -> None:
    clock = LogicalClock()
    zone = BinZone(
        1,
        3,
        size_classes=frozenset({"small", "large"}),
        hazards=frozenset({"general", "cold"}),
    )
    pool = WarehouseBinPool({"A": zone, "B": zone}, clock=clock)
    pool.acquire(
        "A",
        "one",
        ttl=5,
        size_class="small",
        hazard="general",
        request_id="one",
    )
    pool.acquire(
        "B",
        "two",
        ttl=5,
        size_class="large",
        hazard="cold",
        request_id="two",
    )
    checkpoint = pool.checkpoint()
    metadata = {item.scope: item for item in checkpoint.metadata}
    first = metadata["A"]

    with pytest.raises(TypeError, match="BinPoolCheckpoint"):
        WarehouseBinPool.from_checkpoint(object(), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="zone names must be unique"):
        WarehouseBinPool.from_checkpoint(
            replace(checkpoint, zones=checkpoint.zones * 2), clock=clock
        )
    with pytest.raises(TypeError, match="BinZone"):
        WarehouseBinPool.from_checkpoint(
            replace(checkpoint, zones=(("A", object()),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="at least one bin zone"):
        WarehouseBinPool.from_checkpoint(replace(checkpoint, zones=()), clock=clock)
    with pytest.raises(TypeError, match="invalid bin metadata"):
        WarehouseBinPool.from_checkpoint(
            replace(checkpoint, metadata=(object(),)),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="unknown zone"):
        WarehouseBinPool.from_checkpoint(
            replace(checkpoint, metadata=(replace(first, scope="missing"),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="zone policy"):
        WarehouseBinPool.from_checkpoint(
            replace(checkpoint, metadata=(replace(first, size_class="bad"),)),
            clock=clock,
        )
    with pytest.raises(ValueError, match="lease history"):
        WarehouseBinPool.from_checkpoint(
            replace(checkpoint, metadata=(replace(first, token=999),)), clock=clock
        )
    with pytest.raises(ValueError, match="scope/token pairs must be unique"):
        WarehouseBinPool.from_checkpoint(
            replace(checkpoint, metadata=(first, first)), clock=clock
        )
    with pytest.raises(ValueError, match="every bin lease"):
        WarehouseBinPool.from_checkpoint(
            replace(checkpoint, metadata=(first,)), clock=clock
        )


def test_software_policy_checkout_and_renewal_public_edges() -> None:
    clock = LogicalClock()
    with pytest.raises(ValueError, match="at least one product"):
        SoftwareSeatPool({}, clock=clock)
    with pytest.raises(ValueError, match="product"):
        SoftwareSeatPool({"": 1}, clock=clock)
    with pytest.raises(ValueError, match="capacity"):
        SoftwareSeatPool({"ide": 0}, clock=clock)
    with pytest.raises(ValueError, match="entitlement owner"):
        SoftwareSeatPool({"ide": 1}, entitlements={"": {"ide": 1}}, clock=clock)
    with pytest.raises(UnknownProductError):
        SoftwareSeatPool({"ide": 1}, entitlements={"owner": {"other": 1}}, clock=clock)
    with pytest.raises(ValueError, match="entitlement limit"):
        SoftwareSeatPool({"ide": 1}, entitlements={"owner": {"ide": 0}}, clock=clock)

    unrestricted = SoftwareSeatPool({"ide": 1}, clock=clock)
    seat = unrestricted.checkout("ide", "owner", ttl=5, request_id="login")
    with pytest.raises(SeatUnavailableError):
        unrestricted.checkout("ide", "other", ttl=5)
    with pytest.raises(LeaseRequestConflictError):
        unrestricted.checkout("ide", "changed", ttl=5, request_id="login")
    with pytest.raises(UnknownProductError):
        unrestricted.checkout("other", "owner", ttl=1)
    with pytest.raises(ValueError, match="count"):
        unrestricted.checkout("ide", "owner", ttl=1, count=0)
    with pytest.raises(TypeError, match="SeatLease"):
        unrestricted.renew(cast(NumericLease, object()), ttl=1)
    with pytest.raises(UnknownProductError):
        unrestricted.renew(NumericLease("other", seat.lease), ttl=1)
    with pytest.raises(ValueError, match="outside"):
        unrestricted.validate_fence(seat, 2)

    unrestricted.release(seat)
    checkpoint = unrestricted.checkpoint()
    assert not checkpoint.entitlement_restricted
    assert not checkpoint.entitlements
    restored = SoftwareSeatPool.from_checkpoint(checkpoint, clock=clock)
    assert restored.checkout("ide", "next", ttl=1).resource == Span(1, 2)


def test_software_restore_rejects_malformed_entitlement_policy() -> None:
    clock = LogicalClock()
    pool = SoftwareSeatPool({"ide": 2}, entitlements={"alice": {"ide": 2}}, clock=clock)
    pool.checkout("ide", "alice", ttl=5)
    checkpoint = pool.checkpoint()

    with pytest.raises(TypeError, match="SeatPoolCheckpoint"):
        SoftwareSeatPool.from_checkpoint(object(), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="products must be unique"):
        SoftwareSeatPool.from_checkpoint(
            replace(checkpoint, products=checkpoint.products * 2), clock=clock
        )
    with pytest.raises(ValueError, match="at least one product"):
        SoftwareSeatPool.from_checkpoint(replace(checkpoint, products=()), clock=clock)
    with pytest.raises(TypeError, match="must be a bool"):
        SoftwareSeatPool.from_checkpoint(
            replace(checkpoint, entitlement_restricted=1),  # type: ignore[arg-type]
            clock=clock,
        )
    with pytest.raises(ValueError, match="owners must be unique"):
        SoftwareSeatPool.from_checkpoint(
            replace(checkpoint, entitlements=checkpoint.entitlements * 2),
            clock=clock,
        )
    with pytest.raises(ValueError, match="products must be unique"):
        SoftwareSeatPool.from_checkpoint(
            replace(
                checkpoint,
                entitlements=(("alice", (("ide", 2), ("ide", 2))),),
            ),
            clock=clock,
        )
    with pytest.raises(UnknownProductError):
        SoftwareSeatPool.from_checkpoint(
            replace(
                checkpoint,
                entitlements=(("alice", (("other", 2),)),),
            ),
            clock=clock,
        )
    with pytest.raises(ValueError, match="unrestricted"):
        SoftwareSeatPool.from_checkpoint(
            replace(checkpoint, entitlement_restricted=False), clock=clock
        )


def test_software_restricted_renewal_requires_original_entitlement() -> None:
    clock = LogicalClock()
    entitled = SoftwareSeatPool(
        {"ide": 1}, entitlements={"alice": {"ide": 1}}, clock=clock
    )
    seat = entitled.checkout("ide", "alice", ttl=5)
    foreign_policy = SoftwareSeatPool(
        {"ide": 1}, entitlements={"bob": {"ide": 1}}, clock=clock
    )
    foreign_handle = NumericLease(
        "ide",
        replace(seat.lease, pool_id=foreign_policy.snapshot().pools[0][1].pool_id),
    )
    with pytest.raises(EntitlementError):
        foreign_policy.renew(foreign_handle, ttl=1)


def test_vlan_policy_bounds_identity_and_fencing_edges() -> None:
    clock = LogicalClock()
    with pytest.raises(ValueError, match="at least one network scope"):
        VlanTagPool((), clock=clock)
    with pytest.raises(ValueError, match="unique"):
        VlanTagPool(("edge", "edge"), clock=clock)
    with pytest.raises(ValueError, match="network scope"):
        VlanTagPool(("",), clock=clock)
    with pytest.raises(VlanScopeError, match="unknown scopes"):
        VlanTagPool(("edge",), scope_reserved={"other": ((1, 2),)}, clock=clock)
    with pytest.raises(ValueError, match="must not exceed"):
        VlanTagPool(("edge",), reserved_ranges=((2, 1),), clock=clock)
    with pytest.raises(ValueError, match="complete leasing domain"):
        VlanTagPool(("edge",), reserved_ranges=((1, 4094),), clock=clock)

    pool = VlanTagPool(("edge",), reserved_ranges=((10, 11),), clock=clock)
    with pytest.raises(VlanScopeError, match="unknown"):
        pool.acquire("other", "owner", ttl=1)
    with pytest.raises(ValueError, match="count"):
        pool.acquire("edge", "owner", ttl=1, count=0)
    with pytest.raises(ValueError, match="1..4094"):
        pool.acquire("edge", "owner", ttl=1, count=2, start_tag=4094)
    with pytest.raises(VlanUnavailableError):
        pool.acquire("edge", "owner", ttl=1, start_tag=10)

    lease = pool.acquire("edge", "owner", ttl=5, request_id="request")
    with pytest.raises(LeaseRequestConflictError):
        pool.acquire("edge", "changed", ttl=5, request_id="request")
    with pytest.raises(ValueError, match="outside"):
        pool.validate_fence(lease, lease.resource.end)


def test_vlan_restore_rejects_invalid_type_and_out_of_ieee_domain() -> None:
    clock = LogicalClock()
    with pytest.raises(TypeError, match="VlanPoolCheckpoint"):
        VlanTagPool.from_checkpoint(object(), clock=clock)  # type: ignore[arg-type]

    outside = PoolGroup({"edge": (Span(0, 1),)}, clock=clock).checkpoint()
    with pytest.raises(ValueError, match="outside 1..4094"):
        VlanTagPool.from_checkpoint(VlanPoolCheckpoint(outside), clock=clock)


def test_numeric_ip_public_input_exhaustion_and_fence_edges() -> None:
    clock = LogicalClock()
    with pytest.raises(ValueError, match="canonical CIDR"):
        NumericIPAddressPool("192.0.2.1/24", clock=clock)
    with pytest.raises(ValueError, match="canonical CIDR"):
        NumericIPAddressPool(cast(str, object()), clock=clock)
    with pytest.raises(ValueError, match="invalid reserved address"):
        NumericIPAddressPool("192.0.2.0/29", reserved=("bad",), clock=clock)

    pool = NumericIPAddressPool(
        "192.0.2.0/30",
        reserve_network=False,
        reserve_broadcast=False,
        clock=clock,
    )
    with pytest.raises(ValueError, match="count"):
        pool.acquire("owner", ttl=1, count=0)
    with pytest.raises(ValueError, match="invalid start_address"):
        pool.acquire("owner", ttl=1, start_address="bad")
    lease = pool.acquire("owner", ttl=5, count=4, request_id="request")
    assert pool.first_address(lease) == ipaddress.ip_address("192.0.2.0")
    assert pool.last_address(lease) == ipaddress.ip_address("192.0.2.3")
    with pytest.raises(AddressUnavailableError):
        pool.acquire("other", ttl=1)
    with pytest.raises(LeaseRequestConflictError):
        pool.acquire("changed", ttl=5, count=4, request_id="request")
    with pytest.raises(ValueError, match="invalid address"):
        pool.validate_fence(lease, "bad")
    with pytest.raises(ValueError, match="inside"):
        pool.validate_fence(lease, "2001:db8::1")

    partial_pool = NumericIPAddressPool(
        "192.0.2.0/29",
        reserve_network=False,
        reserve_broadcast=False,
        clock=clock,
    )
    partial = partial_pool.acquire("owner", ttl=1, count=2)
    with pytest.raises(ValueError, match="outside the leased block"):
        partial_pool.validate_fence(partial, "192.0.2.2")


def test_numeric_ip_restore_rejects_scope_and_domain_corruption() -> None:
    clock = LogicalClock()
    pool = NumericIPAddressPool("192.0.2.0/29", clock=clock)
    checkpoint = pool.checkpoint()
    with pytest.raises(TypeError, match="AddressPoolCheckpoint"):
        NumericIPAddressPool.from_checkpoint(object(), clock=clock)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canonical CIDR"):
        NumericIPAddressPool.from_checkpoint(
            replace(checkpoint, network="192.0.2.1/29"), clock=clock
        )
    with pytest.raises(ValueError, match="scope does not match"):
        NumericIPAddressPool.from_checkpoint(
            replace(checkpoint, network="198.51.100.0/29"), clock=clock
        )

    network = ipaddress.ip_network("192.0.2.0/29")
    network_start = 3_221_225_984
    outside_group = PoolGroup(
        {network.with_prefixlen: (Span(network_start - 1, network_start),)},
        clock=clock,
    )
    outside_checkpoint = AddressPoolCheckpoint(
        network.with_prefixlen, outside_group.checkpoint()
    )
    with pytest.raises(ValueError, match="outside its network"):
        NumericIPAddressPool.from_checkpoint(outside_checkpoint, clock=clock)
