"""Application scenario manifest and implementation-status registry."""

from treemendous.applications.registry import (
    EXPECTED_FAMILY_COUNTS,
    SCENARIO_SPECS,
    SCENARIOS_BY_ID,
    ScenarioNotFoundError,
    ScenarioNotImplementedError,
    ScenarioRegistryError,
    ScenarioSpec,
    ScenarioStatus,
    create_application,
    get_scenario,
    list_scenarios,
    scenario_status_counts,
    validate_catalog_evidence,
    validate_completion_evidence,
)

__all__ = [
    "EXPECTED_FAMILY_COUNTS",
    "SCENARIO_SPECS",
    "SCENARIOS_BY_ID",
    "ScenarioNotFoundError",
    "ScenarioNotImplementedError",
    "ScenarioRegistryError",
    "ScenarioSpec",
    "ScenarioStatus",
    "create_application",
    "get_scenario",
    "list_scenarios",
    "scenario_status_counts",
    "validate_catalog_evidence",
    "validate_completion_evidence",
]
