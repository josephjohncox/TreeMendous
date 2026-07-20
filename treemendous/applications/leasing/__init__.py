"""Reusable process-local engines for the eight numeric leasing scenarios.

Exports are resolved lazily so importing the data-only family manifest does not
also import every application engine.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from treemendous.applications.leasing.database_ids import (
        CommittedIdBatch as CommittedIdBatch,
    )
    from treemendous.applications.leasing.database_ids import (
        DatabaseIdPool as DatabaseIdPool,
    )
    from treemendous.applications.leasing.game_regions import (
        GameRegionPool as GameRegionPool,
    )
    from treemendous.applications.leasing.numeric_ip_pools import (
        NumericIPAddressPool as NumericIPAddressPool,
    )
    from treemendous.applications.leasing.phone_extensions import (
        PhoneExtensionPool as PhoneExtensionPool,
    )
    from treemendous.applications.leasing.software_seats import (
        SoftwareSeatPool as SoftwareSeatPool,
    )
    from treemendous.applications.leasing.tcp_udp_ports import (
        PortLeaseEngine as PortLeaseEngine,
    )
    from treemendous.applications.leasing.tcp_udp_ports import (
        PortProtocol as PortProtocol,
    )
    from treemendous.applications.leasing.vlan_tags import VlanTagPool as VlanTagPool
    from treemendous.applications.leasing.warehouse_bins import BinZone as BinZone
    from treemendous.applications.leasing.warehouse_bins import (
        WarehouseBinPool as WarehouseBinPool,
    )

_EXPORTS = {
    "BinZone": ("warehouse_bins", "BinZone"),
    "CommittedIdBatch": ("database_ids", "CommittedIdBatch"),
    "DatabaseIdPool": ("database_ids", "DatabaseIdPool"),
    "GameRegionPool": ("game_regions", "GameRegionPool"),
    "NumericIPAddressPool": ("numeric_ip_pools", "NumericIPAddressPool"),
    "PhoneExtensionPool": ("phone_extensions", "PhoneExtensionPool"),
    "PortLeaseEngine": ("tcp_udp_ports", "PortLeaseEngine"),
    "PortProtocol": ("tcp_udp_ports", "PortProtocol"),
    "SoftwareSeatPool": ("software_seats", "SoftwareSeatPool"),
    "VlanTagPool": ("vlan_tags", "VlanTagPool"),
    "WarehouseBinPool": ("warehouse_bins", "WarehouseBinPool"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve one public engine export without eager family imports."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute)
    globals()[name] = value
    return value
