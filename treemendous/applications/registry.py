"""Honest manifest for documented Tree-Mendous application scenarios.

The manifest records implementation evidence.  It never substitutes the
benchmark-only application-shaped traces for real application engines.
"""

from __future__ import annotations

import importlib
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

_SCENARIO_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ScenarioStatus(Enum):
    """Implementation state for one documented scenario."""

    PLANNED = "planned"
    COMPLETE = "complete"


class ScenarioRegistryError(RuntimeError):
    """Base error for the application scenario registry."""


class ScenarioNotFoundError(ScenarioRegistryError, KeyError):
    """Raised when a scenario identifier is absent from the manifest."""


class ScenarioNotImplementedError(ScenarioRegistryError):
    """Raised when a documented scenario has no completed engine."""


def _engine_reference_parts(reference: str) -> tuple[str, str]:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("engine reference must use 'module:callable' syntax")
    return module_name, attribute_name


def _resolve_factory(reference: str) -> Callable[..., Any]:
    module_name, attribute_name = _engine_reference_parts(reference)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"engine reference is not resolvable: {reference}") from exc
    if not callable(factory):
        raise ValueError(f"engine reference is not callable: {reference}")
    return factory


@dataclass(frozen=True)
class ScenarioSpec:
    """Immutable implementation manifest entry for one application scenario."""

    id: str
    title: str
    category: str
    family: str
    description: str
    status: ScenarioStatus = ScenarioStatus.PLANNED
    engine: str | None = None
    example: str | None = None
    oracle: str | None = None
    benchmark: str | None = None
    docs: str | None = None

    def __post_init__(self) -> None:
        if not _SCENARIO_ID.fullmatch(self.id):
            raise ValueError("scenario id must be lowercase kebab-case")
        if self.family not in EXPECTED_FAMILY_COUNTS:
            raise ValueError(f"unknown scenario family: {self.family}")
        if not isinstance(self.status, ScenarioStatus):
            raise TypeError("scenario status must be a ScenarioStatus")
        if not all((self.title, self.category, self.description)):
            raise ValueError("scenario title, category, and description are required")
        if self.status is ScenarioStatus.PLANNED:
            if self.engine is not None:
                raise ValueError("planned scenarios cannot expose an engine factory")
            return
        references = {
            "engine": self.engine,
            "example": self.example,
            "oracle": self.oracle,
            "benchmark": self.benchmark,
            "docs": self.docs,
        }
        missing = tuple(name for name, value in references.items() if not value)
        if missing:
            raise ValueError(
                "complete scenarios require every artifact reference; missing="
                + ",".join(missing)
            )
        engine = self.engine
        if engine is None:
            raise ValueError("complete scenario engine reference is required")
        # Syntax is safe to validate while the registry is importing. Actual
        # factory import and repository-artifact checks are deliberately lazy
        # so completed engines may import this public registry without cycles.
        _engine_reference_parts(engine)


_ARTIFACT_RULES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "example": ("examples/", ".py"),
        "oracle": ("tests/oracles/", ".py"),
        "benchmark": ("tests/performance/", ".py"),
        "docs": ("docs/scenarios/", ".md"),
    }
)


def validate_completion_evidence(spec: ScenarioSpec, *, root: Path) -> None:
    """Validate one COMPLETE row against callable and repository evidence.

    This explicit CI/documentation gate is separate from dataclass construction:
    importing a canonical registry must never eagerly import its engine modules.
    """
    if spec.status is ScenarioStatus.PLANNED:
        return
    engine = spec.engine
    if engine is None:
        raise ValueError("complete scenario engine reference is required")
    _resolve_factory(engine)
    resolved_root = root.resolve()
    for field_name, (prefix, suffix) in _ARTIFACT_RULES.items():
        reference = getattr(spec, field_name)
        assert isinstance(reference, str)
        if not reference.startswith(prefix) or not reference.endswith(suffix):
            raise ValueError(
                f"{field_name} reference must match {prefix}*{suffix}: {reference}"
            )
        path = Path(reference)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{field_name} reference must stay within the repository")
        candidate = (resolved_root / path).resolve()
        if resolved_root not in candidate.parents or not candidate.is_file():
            raise ValueError(f"{field_name} artifact does not exist: {reference}")


def validate_catalog_evidence(
    specs: tuple[ScenarioSpec, ...] | None = None, *, root: Path
) -> None:
    """Validate every completed canonical scenario against its evidence."""
    selected = SCENARIO_SPECS if specs is None else specs
    for spec in selected:
        validate_completion_evidence(spec, root=root)


EXPECTED_FAMILY_COUNTS: Mapping[str, int] = MappingProxyType(
    {
        "partition": 12,
        "scheduling": 12,
        "catalog": 10,
        "allocator": 8,
        "lease": 8,
    }
)


_SCENARIO_ROWS = (
    (
        "distributed-document-search",
        "Distributed document search",
        "distributed_partition",
        "partition",
        "workers claim document-ID bands and return abandoned bands for retry",
    ),
    (
        "distributed-regex-scan",
        "Distributed regular-expression scan",
        "distributed_partition",
        "partition",
        "workers claim byte-offset chunks while overlap checks model boundary halos",
    ),
    (
        "distributed-genetic-search",
        "Distributed genetic search",
        "distributed_partition",
        "partition",
        "the in-process engine claims and commits one serial generation at a time",
    ),
    (
        "distributed-graph-search",
        "Distributed graph frontier search",
        "distributed_partition",
        "partition",
        "the in-process engine accounts for bounded deterministic frontier expansions",
    ),
    (
        "distributed-sat-search",
        "Distributed SAT search",
        "distributed_partition",
        "partition",
        "workers claim ordinal ranges of assignment prefixes",
    ),
    (
        "distributed-fuzzing",
        "Distributed fuzzing",
        "distributed_partition",
        "partition",
        "workers claim generated-input ordinal ranges and retry failed workers",
    ),
    (
        "distributed-hyperparameter-search",
        "Distributed hyperparameter search",
        "distributed_partition",
        "partition",
        "workers claim deterministic trial-ID ranges",
    ),
    (
        "distributed-log-replay",
        "Distributed log replay",
        "distributed_partition",
        "partition",
        "consumers claim log-offset windows and return interrupted windows",
    ),
    (
        "distributed-build-sharding",
        "Distributed build and test sharding",
        "distributed_partition",
        "partition",
        "runners claim ordered source-file or test-ID ranges",
    ),
    (
        "map-reduce-input-splits",
        "Map-reduce input splitting",
        "distributed_partition",
        "partition",
        "workers claim byte and record ranges from partitioned inputs",
    ),
    (
        "distributed-web-crawl",
        "Distributed web crawl",
        "distributed_partition",
        "partition",
        "the in-process engine accounts for one normalized-URL fetch at a time",
    ),
    (
        "distributed-index-merge",
        "Distributed search-index merge",
        "distributed_partition",
        "partition",
        "workers claim term-ID and posting-list bands during index merges",
    ),
    (
        "distributed-cluster-scheduling",
        "Distributed cluster scheduling",
        "scheduling_reservation",
        "scheduling",
        "jobs reserve bounded compute-lane windows with deadlines and cancellation",
    ),
    (
        "gpu-stream-scheduling",
        "GPU stream scheduling",
        "scheduling_reservation",
        "scheduling",
        "kernels reserve stream-time windows under occupancy pressure",
    ),
    (
        "render-farm-frames",
        "Render-farm frame scheduling",
        "scheduling_reservation",
        "scheduling",
        "workers reserve contiguous frame ranges and return cancelled renders",
    ),
    (
        "ci-runner-reservations",
        "CI runner reservations",
        "scheduling_reservation",
        "scheduling",
        "jobs reserve runner-time windows with release times and deadlines",
    ),
    (
        "meeting-room-booking",
        "Meeting-room booking",
        "scheduling_reservation",
        "scheduling",
        "rooms expose available time ranges for bounded reservations",
    ),
    (
        "airline-gate-assignment",
        "Airline gate assignment",
        "scheduling_reservation",
        "scheduling",
        "flights reserve gate-time windows with turnaround constraints",
    ),
    (
        "operating-room-booking",
        "Operating-room booking",
        "scheduling_reservation",
        "scheduling",
        "procedures reserve room-time ranges and cancellations restore capacity",
    ),
    (
        "laboratory-equipment-booking",
        "Laboratory equipment booking",
        "scheduling_reservation",
        "scheduling",
        "experiments reserve instrument-time windows",
    ),
    (
        "fleet-charging-windows",
        "Fleet charging windows",
        "scheduling_reservation",
        "scheduling",
        "vehicles reserve charger-time ranges under capacity pressure",
    ),
    (
        "radio-spectrum-timeslots",
        "Radio spectrum timeslots",
        "scheduling_reservation",
        "scheduling",
        "transmitters reserve channel-time windows",
    ),
    (
        "warehouse-dock-appointments",
        "Warehouse dock appointments",
        "scheduling_reservation",
        "scheduling",
        "carriers reserve dock-time windows and cancel stale appointments",
    ),
    (
        "maintenance-window-planning",
        "Maintenance-window planning",
        "scheduling_reservation",
        "scheduling",
        "services reserve maintenance windows within allowed periods",
    ),
    (
        "genomic-annotation-overlap",
        "Genomic annotation overlap",
        "overlap_catalog",
        "catalog",
        "unioned gene, exon, variant, and read coverage is queried by overlap",
    ),
    (
        "source-diagnostic-ranges",
        "Source diagnostic ranges",
        "overlap_catalog",
        "catalog",
        "unioned diagnostic regions are queried by byte or token span",
    ),
    (
        "filesystem-byte-locks",
        "Filesystem byte-range locks",
        "overlap_catalog",
        "catalog",
        "lock requests query existing byte-range intersections",
    ),
    (
        "database-key-range-locks",
        "Database key-range locks",
        "overlap_catalog",
        "catalog",
        "transactions query normalized integer key bands for conflicts",
    ),
    (
        "packet-sequence-reassembly",
        "Packet sequence reassembly",
        "overlap_catalog",
        "catalog",
        "received sequence-number ranges are queried for overlap and gaps",
    ),
    (
        "subtitle-cue-ranges",
        "Subtitle cue ranges",
        "overlap_catalog",
        "catalog",
        "time-coded cues are queried at playback positions and edit windows",
    ),
    (
        "video-edit-regions",
        "Video edit regions",
        "overlap_catalog",
        "catalog",
        "frame ranges are queried for cuts, effects, and render invalidation",
    ),
    (
        "timeseries-alert-windows",
        "Time-series alert windows",
        "overlap_catalog",
        "catalog",
        "alert and suppression windows are queried by timestamp range",
    ),
    (
        "distributed-trace-spans",
        "Distributed trace spans",
        "overlap_catalog",
        "catalog",
        "normalized timestamp ranges locate concurrent trace activity",
    ),
    (
        "morton-geospatial-ranges",
        "Morton-code geospatial ranges",
        "overlap_catalog",
        "catalog",
        "one-dimensional Morton-code bands approximate spatial query regions",
    ),
    (
        "heap-free-space",
        "Heap free-space allocation",
        "allocation_churn",
        "allocator",
        "allocators split, merge, reserve, and release address ranges",
    ),
    (
        "disk-block-allocation",
        "Disk block allocation",
        "allocation_churn",
        "allocator",
        "filesystems reserve and release contiguous block ranges",
    ),
    (
        "virtual-address-space",
        "Virtual address-space management",
        "allocation_churn",
        "allocator",
        "mappings reserve and release page-aligned virtual address ranges",
    ),
    (
        "database-page-allocation",
        "Database page allocation",
        "allocation_churn",
        "allocator",
        "storage engines allocate and recycle page-ID ranges",
    ),
    (
        "object-store-multipart-ranges",
        "Object-store multipart ranges",
        "allocation_churn",
        "allocator",
        "uploads track completed, missing, and retried byte ranges",
    ),
    (
        "cdn-byte-range-cache",
        "CDN byte-range cache",
        "allocation_churn",
        "allocator",
        "cache entries track resident and evicted object byte ranges",
    ),
    (
        "gpu-memory-arena",
        "GPU memory arena",
        "allocation_churn",
        "allocator",
        "buffers reserve and release device-address ranges",
    ),
    (
        "ring-buffer-sequences",
        "Ring-buffer sequence capacity",
        "allocation_churn",
        "allocator",
        "producers reserve sequence-number ranges and consumers release them",
    ),
    (
        "tcp-udp-port-leases",
        "TCP and UDP port leases",
        "resource_leasing",
        "lease",
        "services claim contiguous port ranges and return expired leases",
    ),
    (
        "numeric-ip-address-pools",
        "Numeric IP address pools",
        "resource_leasing",
        "lease",
        "address managers claim integer-encoded address ranges",
    ),
    (
        "database-id-pools",
        "Database ID pools",
        "resource_leasing",
        "lease",
        "writers claim contiguous identifier batches",
    ),
    (
        "software-license-seats",
        "Software license seats",
        "resource_leasing",
        "lease",
        "clients claim seat-ID ranges and return expired sessions",
    ),
    (
        "warehouse-bin-ranges",
        "Warehouse bin ranges",
        "resource_leasing",
        "lease",
        "inventory jobs claim contiguous normalized bin-ID ranges",
    ),
    (
        "game-world-region-ids",
        "Game-world region IDs",
        "resource_leasing",
        "lease",
        "servers claim region-ID bands and transfer ownership",
    ),
    (
        "vlan-tag-pools",
        "VLAN tag pools",
        "resource_leasing",
        "lease",
        "network controllers lease contiguous VLAN identifier ranges",
    ),
    (
        "phone-extension-pools",
        "Phone extension pools",
        "resource_leasing",
        "lease",
        "provisioning systems claim extension-number ranges",
    ),
)

_FAMILY_EVIDENCE_MODULES: Mapping[str, str] = MappingProxyType(
    {
        "partition": "treemendous.applications.partitioning.manifest",
        "scheduling": "treemendous.applications.scheduling.manifest",
        "catalog": "treemendous.applications.catalogs.manifest",
        "allocator": "treemendous.applications.allocation.manifest",
        "lease": "treemendous.applications.leasing.manifest",
    }
)
_EVIDENCE_FIELDS = frozenset({"engine", "example", "oracle", "benchmark", "docs"})


def _load_implementation_evidence() -> dict[str, dict[str, str]]:
    """Load data-only family manifests without importing application engines."""
    collected: dict[str, dict[str, str]] = {}
    rows_by_id = {row[0]: row for row in _SCENARIO_ROWS}
    for family, module_name in _FAMILY_EVIDENCE_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = exc.name if exc.name is not None else ""
            if missing == module_name:
                continue
            if missing and module_name.startswith(f"{missing}."):
                continue
            raise
        raw = getattr(module, "EVIDENCE", None)
        if not isinstance(raw, Mapping):
            raise TypeError(f"{module_name}.EVIDENCE must be a mapping")
        for scenario_id, references in raw.items():
            if not isinstance(scenario_id, str) or scenario_id not in rows_by_id:
                raise ValueError(f"unknown scenario evidence id: {scenario_id!r}")
            if rows_by_id[scenario_id][3] != family:
                raise ValueError(f"scenario evidence has wrong family: {scenario_id}")
            if scenario_id in collected:
                raise ValueError(f"duplicate scenario evidence: {scenario_id}")
            if not isinstance(references, Mapping):
                raise TypeError(f"scenario evidence must be a mapping: {scenario_id}")
            if set(references) != _EVIDENCE_FIELDS:
                raise ValueError(
                    f"scenario evidence fields differ for {scenario_id}: "
                    f"{set(references)!r}"
                )
            if not all(
                isinstance(value, str) and value for value in references.values()
            ):
                raise ValueError(
                    "scenario evidence references must be nonempty strings"
                )
            collected[scenario_id] = dict(references)
    return collected


_IMPLEMENTATION_EVIDENCE = _load_implementation_evidence()
SCENARIO_SPECS = tuple(
    ScenarioSpec(
        *row,
        status=(
            ScenarioStatus.COMPLETE
            if row[0] in _IMPLEMENTATION_EVIDENCE
            else ScenarioStatus.PLANNED
        ),
        **_IMPLEMENTATION_EVIDENCE.get(row[0], {}),
    )
    for row in _SCENARIO_ROWS
)


def _validate_catalog(specs: tuple[ScenarioSpec, ...]) -> None:
    identifiers = tuple(spec.id for spec in specs)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("scenario ids must be unique")
    counts = Counter(spec.family for spec in specs)
    if counts != Counter(EXPECTED_FAMILY_COUNTS):
        raise ValueError(
            f"scenario family counts differ: {dict(counts)!r} "
            f"!= {dict(EXPECTED_FAMILY_COUNTS)!r}"
        )


_validate_catalog(SCENARIO_SPECS)
SCENARIOS_BY_ID: Mapping[str, ScenarioSpec] = MappingProxyType(
    {spec.id: spec for spec in SCENARIO_SPECS}
)


def list_scenarios(
    *,
    status: ScenarioStatus | None = None,
    family: str | None = None,
) -> tuple[ScenarioSpec, ...]:
    """Return manifest entries in canonical documentation order."""
    if family is not None and family not in EXPECTED_FAMILY_COUNTS:
        raise ValueError(f"unknown scenario family: {family}")
    return tuple(
        spec
        for spec in SCENARIO_SPECS
        if (status is None or spec.status is status)
        and (family is None or spec.family == family)
    )


def get_scenario(scenario_id: str) -> ScenarioSpec:
    """Return one manifest entry or raise a registry-specific error."""
    try:
        return SCENARIOS_BY_ID[scenario_id]
    except KeyError:
        raise ScenarioNotFoundError(scenario_id) from None


def scenario_status_counts() -> Mapping[ScenarioStatus, int]:
    """Return immutable completion counts for the current manifest."""
    counts = Counter(spec.status for spec in SCENARIO_SPECS)
    return MappingProxyType({status: counts[status] for status in ScenarioStatus})


def create_application(scenario_id: str, **kwargs: Any) -> Any:
    """Construct a completed engine; planned entries always fail explicitly."""
    spec = get_scenario(scenario_id)
    if spec.status is not ScenarioStatus.COMPLETE or spec.engine is None:
        raise ScenarioNotImplementedError(
            f"scenario {scenario_id!r} is planned and has no application engine"
        )
    return _resolve_factory(spec.engine)(**kwargs)
