"""Implementation evidence for exactly the eight numeric lease scenarios."""

from __future__ import annotations

from types import MappingProxyType


def _evidence(module: str, stem: str) -> MappingProxyType[str, str]:
    return MappingProxyType(
        {
            "engine": f"treemendous.applications.leasing.{module}:create_engine",
            "example": f"examples/applications/leasing/{stem}.py",
            "oracle": f"tests/oracles/applications/leasing/{stem}.py",
            "benchmark": f"tests/performance/applications/leasing/{stem}.py",
            "docs": f"docs/scenarios/leasing/{stem}.md",
        }
    )


EVIDENCE = MappingProxyType(
    {
        "tcp-udp-port-leases": _evidence("tcp_udp_ports", "tcp_udp_port_leases"),
        "numeric-ip-address-pools": _evidence(
            "numeric_ip_pools", "numeric_ip_address_pools"
        ),
        "database-id-pools": _evidence("database_ids", "database_id_pools"),
        "software-license-seats": _evidence(
            "software_seats", "software_license_seats"
        ),
        "warehouse-bin-ranges": _evidence(
            "warehouse_bins", "warehouse_bin_ranges"
        ),
        "game-world-region-ids": _evidence(
            "game_regions", "game_world_region_ids"
        ),
        "vlan-tag-pools": _evidence("vlan_tags", "vlan_tag_pools"),
        "phone-extension-pools": _evidence(
            "phone_extensions", "phone_extension_pools"
        ),
    }
)
