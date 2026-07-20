from __future__ import annotations

from treemendous.applications.scheduling.manifest import EVIDENCE

EXPECTED_IDS = frozenset(
    {
        "distributed-cluster-scheduling",
        "gpu-stream-scheduling",
        "render-farm-frames",
        "ci-runner-reservations",
        "meeting-room-booking",
        "airline-gate-assignment",
        "operating-room-booking",
        "laboratory-equipment-booking",
        "fleet-charging-windows",
        "radio-spectrum-timeslots",
        "warehouse-dock-appointments",
        "maintenance-window-planning",
    }
)


def test_manifest_exactly_covers_scheduling_family() -> None:
    assert frozenset(EVIDENCE) == EXPECTED_IDS
    assert len(EVIDENCE) == 12
    for references in EVIDENCE.values():
        assert frozenset(references) == frozenset(
            {"engine", "example", "oracle", "benchmark", "docs"}
        )
