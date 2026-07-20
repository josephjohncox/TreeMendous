"""Focused contracts for correctness-attested leasing benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature

import pytest

from tests.performance.applications.harness import ApplicationSample
from tests.performance.applications.leasing import (
    database_id_pools,
    game_world_region_ids,
    numeric_ip_address_pools,
    phone_extension_pools,
    software_license_seats,
    tcp_udp_port_leases,
    vlan_tag_pools,
    warehouse_bin_ranges,
)
from tests.performance.applications.leasing._shared import MAX_OPERATIONS

Runner = Callable[..., ApplicationSample]

_RUNNERS: tuple[tuple[str, Runner], ...] = (
    ("database-id-pools", database_id_pools.run_benchmark),
    ("game-world-region-ids", game_world_region_ids.run_benchmark),
    ("numeric-ip-address-pools", numeric_ip_address_pools.run_benchmark),
    ("phone-extension-pools", phone_extension_pools.run_benchmark),
    ("software-license-seats", software_license_seats.run_benchmark),
    ("tcp-udp-port-leases", tcp_udp_port_leases.run_benchmark),
    ("vlan-tag-pools", vlan_tag_pools.run_benchmark),
    ("warehouse-bin-ranges", warehouse_bin_ranges.run_benchmark),
)


@pytest.mark.benchmark
@pytest.mark.parametrize(("scenario_id", "runner"), _RUNNERS)
def test_leasing_benchmark_attests_full_lifecycle_deterministically(
    scenario_id: str, runner: Runner
) -> None:
    first = runner(operations=8, seed=17)
    repeated = runner(operations=8, seed=17)

    assert isinstance(first, ApplicationSample)
    assert first.scenario_id == scenario_id
    assert first.operations == 8
    assert first.validated
    assert first.execution_ns >= 0
    assert first.result_checksum == repeated.result_checksum
    assert first.state_checksum == repeated.state_checksum
    assert first.counters_checksum == repeated.counters_checksum
    assert first.evidence_checksum == repeated.evidence_checksum


@pytest.mark.parametrize(("scenario_id", "runner"), _RUNNERS)
def test_leasing_benchmark_has_uniform_bounded_entrypoint(
    scenario_id: str, runner: Runner
) -> None:
    del scenario_id
    parameters = signature(runner).parameters
    assert tuple(parameters) == ("operations", "seed")

    with pytest.raises(ValueError, match="positive"):
        runner(operations=0, seed=42)
    with pytest.raises(ValueError, match="exceed"):
        runner(operations=MAX_OPERATIONS + 1, seed=42)
    with pytest.raises(TypeError, match="seed"):
        runner(operations=1, seed=True)
