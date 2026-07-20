"""Reusable deterministic in-memory scheduling and reservation engines.

Exports are resolved lazily so importing the data-only family manifest does not
also import every application engine.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from treemendous.applications.scheduling.airline_gates import (
        AirlineGateScheduler as AirlineGateScheduler,
    )
    from treemendous.applications.scheduling.airline_gates import (
        create_airline_gate_scheduler as create_airline_gate_scheduler,
    )
    from treemendous.applications.scheduling.ci_runners import (
        CIRunnerScheduler as CIRunnerScheduler,
    )
    from treemendous.applications.scheduling.ci_runners import (
        create_ci_runner_scheduler as create_ci_runner_scheduler,
    )
    from treemendous.applications.scheduling.cluster import (
        ClusterScheduler as ClusterScheduler,
    )
    from treemendous.applications.scheduling.cluster import (
        create_cluster_scheduler as create_cluster_scheduler,
    )
    from treemendous.applications.scheduling.fleet_charging import (
        FleetChargingScheduler as FleetChargingScheduler,
    )
    from treemendous.applications.scheduling.fleet_charging import (
        create_fleet_charging_scheduler as create_fleet_charging_scheduler,
    )
    from treemendous.applications.scheduling.gpu_streams import (
        GPUStreamScheduler as GPUStreamScheduler,
    )
    from treemendous.applications.scheduling.gpu_streams import (
        create_gpu_stream_scheduler as create_gpu_stream_scheduler,
    )
    from treemendous.applications.scheduling.lab_instruments import (
        LabInstrumentScheduler as LabInstrumentScheduler,
    )
    from treemendous.applications.scheduling.lab_instruments import (
        create_lab_instrument_scheduler as create_lab_instrument_scheduler,
    )
    from treemendous.applications.scheduling.maintenance import (
        MaintenanceScheduler as MaintenanceScheduler,
    )
    from treemendous.applications.scheduling.maintenance import (
        create_maintenance_scheduler as create_maintenance_scheduler,
    )
    from treemendous.applications.scheduling.meeting_rooms import (
        MeetingRoomScheduler as MeetingRoomScheduler,
    )
    from treemendous.applications.scheduling.meeting_rooms import (
        create_meeting_room_scheduler as create_meeting_room_scheduler,
    )
    from treemendous.applications.scheduling.operating_rooms import (
        OperatingRoomScheduler as OperatingRoomScheduler,
    )
    from treemendous.applications.scheduling.operating_rooms import (
        create_operating_room_scheduler as create_operating_room_scheduler,
    )
    from treemendous.applications.scheduling.radio_spectrum import (
        RadioSpectrumScheduler as RadioSpectrumScheduler,
    )
    from treemendous.applications.scheduling.radio_spectrum import (
        create_radio_spectrum_scheduler as create_radio_spectrum_scheduler,
    )
    from treemendous.applications.scheduling.render_farm import (
        RenderFarmScheduler as RenderFarmScheduler,
    )
    from treemendous.applications.scheduling.render_farm import (
        create_render_farm_scheduler as create_render_farm_scheduler,
    )
    from treemendous.applications.scheduling.warehouse_docks import (
        WarehouseDockScheduler as WarehouseDockScheduler,
    )
    from treemendous.applications.scheduling.warehouse_docks import (
        create_warehouse_dock_scheduler as create_warehouse_dock_scheduler,
    )

_EXPORTS = {
    "AirlineGateScheduler": ("airline_gates", "AirlineGateScheduler"),
    "CIRunnerScheduler": ("ci_runners", "CIRunnerScheduler"),
    "ClusterScheduler": ("cluster", "ClusterScheduler"),
    "FleetChargingScheduler": ("fleet_charging", "FleetChargingScheduler"),
    "GPUStreamScheduler": ("gpu_streams", "GPUStreamScheduler"),
    "LabInstrumentScheduler": ("lab_instruments", "LabInstrumentScheduler"),
    "MaintenanceScheduler": ("maintenance", "MaintenanceScheduler"),
    "MeetingRoomScheduler": ("meeting_rooms", "MeetingRoomScheduler"),
    "OperatingRoomScheduler": ("operating_rooms", "OperatingRoomScheduler"),
    "RadioSpectrumScheduler": ("radio_spectrum", "RadioSpectrumScheduler"),
    "RenderFarmScheduler": ("render_farm", "RenderFarmScheduler"),
    "WarehouseDockScheduler": ("warehouse_docks", "WarehouseDockScheduler"),
    "create_airline_gate_scheduler": (
        "airline_gates",
        "create_airline_gate_scheduler",
    ),
    "create_ci_runner_scheduler": ("ci_runners", "create_ci_runner_scheduler"),
    "create_cluster_scheduler": ("cluster", "create_cluster_scheduler"),
    "create_fleet_charging_scheduler": (
        "fleet_charging",
        "create_fleet_charging_scheduler",
    ),
    "create_gpu_stream_scheduler": (
        "gpu_streams",
        "create_gpu_stream_scheduler",
    ),
    "create_lab_instrument_scheduler": (
        "lab_instruments",
        "create_lab_instrument_scheduler",
    ),
    "create_maintenance_scheduler": (
        "maintenance",
        "create_maintenance_scheduler",
    ),
    "create_meeting_room_scheduler": (
        "meeting_rooms",
        "create_meeting_room_scheduler",
    ),
    "create_operating_room_scheduler": (
        "operating_rooms",
        "create_operating_room_scheduler",
    ),
    "create_radio_spectrum_scheduler": (
        "radio_spectrum",
        "create_radio_spectrum_scheduler",
    ),
    "create_render_farm_scheduler": (
        "render_farm",
        "create_render_farm_scheduler",
    ),
    "create_warehouse_dock_scheduler": (
        "warehouse_docks",
        "create_warehouse_dock_scheduler",
    ),
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
