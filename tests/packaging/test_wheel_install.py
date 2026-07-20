from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest

from scripts.verify_artifact_contents import verify_wheel

pytestmark = pytest.mark.packaging


def _wheel_from_environment() -> Path:
    configured = os.environ.get("TREEMENDOUS_WHEEL")
    if not configured:
        pytest.skip("set TREEMENDOUS_WHEEL to run clean-install artifact smoke")
    wheel = Path(configured).resolve()
    if not wheel.is_file():
        pytest.fail(f"TREEMENDOUS_WHEEL does not exist: {wheel}")
    return wheel


def _venv_python(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts/python.exe"
    return environment / "bin/python"


def test_wheel_clean_install_and_arbitrary_cwd(tmp_path: Path) -> None:
    wheel = _wheel_from_environment()
    verify_wheel(wheel)
    environment = tmp_path / "venv"
    unrelated = tmp_path / "unrelated-cwd"
    unrelated.mkdir()
    venv.EnvBuilder(with_pip=True).create(environment)
    python = _venv_python(environment)
    subprocess.run(
        [str(python), "-m", "pip", "install", str(wheel)],
        cwd=unrelated,
        check=True,
    )
    required_modules = (
        "treemendous.cpp.boundary",
        "treemendous.cpp.treap",
        "treemendous.cpp.boundary_summary",
        "treemendous.cpp.boundary_summary_optimized",
    )
    clean_environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    for module_name in required_modules:
        subprocess.run(
            [
                str(python),
                "-c",
                f"import importlib; importlib.import_module({module_name!r})",
            ],
            cwd=unrelated,
            env=clean_environment,
            check=True,
        )

    code = """
import json
from treemendous import Span, create_range_set
ranges = create_range_set((0, 100), backend='cpp_boundary')
result = ranges.discard(Span(20, 30), require_covered=True)
assert result.changed_length == 10
assert ranges.first_fit(10, not_before=20).start == 30
print(json.dumps({'backend': 'cpp_boundary', 'free': ranges.snapshot().total_free}))
"""
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=unrelated,
        env=clean_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        smoke_output = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"wheel smoke produced invalid JSON: {result.stdout!r}: {exc}")
    assert smoke_output["free"] == 90

    application_code = """
import importlib
import json
import treemendous
from treemendous.applications import (
    SCENARIO_SPECS,
    ScenarioStatus,
    create_application,
)

expected_root_all = [
    'AvailabilityStats',
    'BackendDecision',
    'BackendInvalidError',
    'BackendRegistry',
    'BackendRequest',
    'BackendSpec',
    'BackendUnavailableError',
    'Capability',
    'DomainInput',
    'IntervalResult',
    'JoinPayloadPolicy',
    'ManagedDomain',
    'ManagedDomainRequiredError',
    'MutationResult',
    'OrderedPayloadPolicy',
    'PayloadPolicy',
    'RangeSet',
    'RangeSetProtocol',
    'RangeSnapshot',
    'Span',
    'UniformPayloadPolicy',
    'create_range_set',
]
assert treemendous.__all__ == expected_root_all
assert len(SCENARIO_SPECS) == 50
assert all(spec.status is ScenarioStatus.COMPLETE for spec in SCENARIO_SPECS)

resolved = []
for spec in SCENARIO_SPECS:
    assert spec.engine is not None
    module_name, separator, attribute_name = spec.engine.partition(':')
    assert separator == ':'
    factory = getattr(importlib.import_module(module_name), attribute_name)
    assert callable(factory)
    resolved.append(spec.id)

representatives = {
    'partition': create_application('distributed-document-search'),
    'scheduling': create_application('distributed-cluster-scheduling'),
    'catalog': create_application('genomic-annotation-overlap'),
    'allocator': create_application('heap-free-space', capacity=128),
    'lease': create_application('tcp-udp-port-leases'),
}
expected_module_families = {
    'partition': '.partitioning.',
    'scheduling': '.scheduling.',
    'catalog': '.catalogs.',
    'allocator': '.allocation.',
    'lease': '.leasing.',
}
for family, application in representatives.items():
    assert expected_module_families[family] in type(application).__module__

print(json.dumps({
    'complete': len(SCENARIO_SPECS),
    'families_constructed': sorted(representatives),
    'resolved': len(resolved),
    'root_all': treemendous.__all__,
}))
"""
    application_result = subprocess.run(
        [str(python), "-c", application_code],
        cwd=unrelated,
        env=clean_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        application_output = json.loads(application_result.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            "wheel application smoke produced invalid JSON: "
            f"{application_result.stdout!r}: {exc}"
        )
    assert application_output == {
        "complete": 50,
        "families_constructed": [
            "allocator",
            "catalog",
            "lease",
            "partition",
            "scheduling",
        ],
        "resolved": 50,
        "root_all": [
            "AvailabilityStats",
            "BackendDecision",
            "BackendInvalidError",
            "BackendRegistry",
            "BackendRequest",
            "BackendSpec",
            "BackendUnavailableError",
            "Capability",
            "DomainInput",
            "IntervalResult",
            "JoinPayloadPolicy",
            "ManagedDomain",
            "ManagedDomainRequiredError",
            "MutationResult",
            "OrderedPayloadPolicy",
            "PayloadPolicy",
            "RangeSet",
            "RangeSetProtocol",
            "RangeSnapshot",
            "Span",
            "UniformPayloadPolicy",
            "create_range_set",
        ],
    }

    readme = Path(__file__).resolve().parents[2] / "README.md"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), flags=re.DOTALL)
    if not blocks:
        pytest.fail("README must contain an executable Python quickstart")
    for index, block in enumerate(blocks):
        completed = subprocess.run(
            [str(python), "-c", block],
            cwd=unrelated,
            env=clean_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.fail(
                f"wheel README Python block {index} failed:\n{completed.stderr}"
            )

    example = readme.parent / "examples/basic_rangeset.py"
    completed = subprocess.run(
        [str(python), str(example)],
        cwd=unrelated,
        env=clean_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "allocated [9, 11)"

    application_example = (
        readme.parent / "examples/applications/partitioning/document_search.py"
    )
    completed = subprocess.run(
        [str(python), str(application_example)],
        cwd=unrelated,
        env=clean_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "document-search: 1 hit"


def test_metal_wheel_resource_and_device_from_arbitrary_cwd(tmp_path: Path) -> None:
    wheel = _wheel_from_environment()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    extension_present = any(
        name.startswith("treemendous/cpp/metal/boundary_summary_metal.")
        and name.endswith((".so", ".pyd"))
        for name in names
    )
    if not extension_present:
        if os.environ.get("TREEMENDOUS_REQUIRE_METAL") == "1":
            pytest.fail(
                "Metal package lane produced a wheel without the Metal extension"
            )
        pytest.skip("wheel does not declare the experimental Metal extension")

    resource = "treemendous/cpp/metal/resources/boundary_summary_metal.metallib"
    assert resource in names
    environment = tmp_path / "metal-venv"
    unrelated = tmp_path / "metal-unrelated-cwd"
    unrelated.mkdir()
    venv.EnvBuilder(with_pip=True).create(environment)
    python = _venv_python(environment)
    subprocess.run(
        [str(python), "-m", "pip", "install", str(wheel)],
        cwd=unrelated,
        check=True,
    )
    code = """
from treemendous.cpp.metal.boundary_summary_metal import MetalBoundarySummaryManager
manager = MetalBoundarySummaryManager()
manager.release_interval(0, 100)
manager.batch_reserve([(10, 20), (30, 40)])
assert manager.get_intervals() == [(0, 10), (20, 30), (40, 100)]
cpu = manager.compute_summary_cpu()
gpu = manager.compute_summary_gpu()
for field in (
    'total_free_length', 'total_occupied_length', 'interval_count',
    'largest_interval_length', 'largest_interval_start',
    'smallest_interval_length', 'total_gaps', 'earliest_start', 'latest_end',
):
    assert getattr(gpu, field) == getattr(cpu, field), field
assert manager.find_best_fit_gpu(10, True) == (0, 10)
assert manager.find_best_fit_gpu(61, True) is None
before = manager.get_intervals()
try:
    manager.batch_release([(50, 60), (70, 70)])
except ValueError as exc:
    assert 'start must be less than end' in str(exc)
else:
    raise AssertionError('invalid Metal batch was accepted')
assert manager.get_intervals() == before
"""
    subprocess.run(
        [str(python), "-c", code],
        cwd=unrelated,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        check=True,
    )
