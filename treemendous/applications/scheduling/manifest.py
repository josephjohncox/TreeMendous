"""Implementation evidence for exactly the twelve scheduling scenarios."""

from __future__ import annotations

from types import MappingProxyType


def _evidence(module: str, factory: str, scenario_id: str) -> dict[str, str]:
    return {
        "engine": f"treemendous.applications.scheduling.{module}:{factory}",
        "example": (
            "examples/one_dimensional/applications/scheduling/"
            f"{module}.py"
        ),
        "oracle": f"tests/oracles/applications/scheduling/{module}.py",
        "benchmark": (
            "tests/performance/applications/scheduling/"
            f"test_{module}_smoke.py"
        ),
        "docs": f"docs/scenarios/scheduling/{scenario_id}.md",
    }


EVIDENCE = MappingProxyType(
    {
        "distributed-cluster-scheduling": _evidence(
            "cluster", "create_cluster_scheduler", "distributed-cluster-scheduling"
        ),
        "gpu-stream-scheduling": _evidence(
            "gpu_streams", "create_gpu_stream_scheduler", "gpu-stream-scheduling"
        ),
        "render-farm-frames": _evidence(
            "render_farm", "create_render_farm_scheduler", "render-farm-frames"
        ),
        "ci-runner-reservations": _evidence(
            "ci_runners", "create_ci_runner_scheduler", "ci-runner-reservations"
        ),
        "meeting-room-booking": _evidence(
            "meeting_rooms", "create_meeting_room_scheduler", "meeting-room-booking"
        ),
        "airline-gate-assignment": _evidence(
            "airline_gates", "create_airline_gate_scheduler", "airline-gate-assignment"
        ),
        "operating-room-booking": _evidence(
            "operating_rooms",
            "create_operating_room_scheduler",
            "operating-room-booking",
        ),
        "laboratory-equipment-booking": _evidence(
            "lab_instruments",
            "create_lab_instrument_scheduler",
            "laboratory-equipment-booking",
        ),
        "fleet-charging-windows": _evidence(
            "fleet_charging",
            "create_fleet_charging_scheduler",
            "fleet-charging-windows",
        ),
        "radio-spectrum-timeslots": _evidence(
            "radio_spectrum",
            "create_radio_spectrum_scheduler",
            "radio-spectrum-timeslots",
        ),
        "warehouse-dock-appointments": _evidence(
            "warehouse_docks",
            "create_warehouse_dock_scheduler",
            "warehouse-dock-appointments",
        ),
        "maintenance-window-planning": _evidence(
            "maintenance",
            "create_maintenance_scheduler",
            "maintenance-window-planning",
        ),
    }
)
