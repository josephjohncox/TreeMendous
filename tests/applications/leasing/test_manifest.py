"""Family-local evidence contracts for exactly eight lease scenarios."""

from __future__ import annotations

from pathlib import Path

from treemendous.applications import (
    ScenarioStatus,
    create_application,
    list_scenarios,
    validate_catalog_evidence,
)
from treemendous.applications.leasing.manifest import EVIDENCE

LEASE_IDS = (
    "tcp-udp-port-leases",
    "numeric-ip-address-pools",
    "database-id-pools",
    "software-license-seats",
    "warehouse-bin-ranges",
    "game-world-region-ids",
    "vlan-tag-pools",
    "phone-extension-pools",
)


def test_manifest_covers_exactly_the_eight_canonical_lease_ids() -> None:
    assert tuple(EVIDENCE) == LEASE_IDS
    specs = list_scenarios(family="lease")
    assert tuple(spec.id for spec in specs) == LEASE_IDS
    assert all(spec.status is ScenarioStatus.COMPLETE for spec in specs)
    validate_catalog_evidence(specs, root=Path(__file__).parents[3])


def test_every_lease_manifest_factory_constructs_its_real_engine() -> None:
    engines = tuple(create_application(scenario_id) for scenario_id in LEASE_IDS)
    expected_names = [
        "PortLeaseEngine",
        "NumericIPAddressPool",
        "DatabaseIdPool",
        "SoftwareSeatPool",
        "WarehouseBinPool",
        "GameRegionPool",
        "VlanTagPool",
        "PhoneExtensionPool",
    ]
    assert [type(engine).__name__ for engine in engines] == expected_names
