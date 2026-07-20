"""Runtime contracts for application-family lazy exports."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FAMILY_EXPORTS = (
    (
        "treemendous.applications.allocation",
        "treemendous.applications.allocation.heap",
        "HeapAllocator",
    ),
    (
        "treemendous.applications.leasing",
        "treemendous.applications.leasing.tcp_udp_ports",
        "PortProtocol",
    ),
    (
        "treemendous.applications.partitioning",
        "treemendous.applications.partitioning.document_search",
        "SearchHit",
    ),
    (
        "treemendous.applications.scheduling",
        "treemendous.applications.scheduling.cluster",
        "ClusterScheduler",
    ),
)


@pytest.mark.parametrize(("package_name", "engine_module", "export"), FAMILY_EXPORTS)
def test_family_export_is_lazy_cached_visible_to_dir_and_rejects_unknown_attrs(
    package_name: str,
    engine_module: str,
    export: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, package_name, raising=False)
    monkeypatch.delitem(sys.modules, engine_module, raising=False)
    package = importlib.import_module(package_name)

    assert not package.TYPE_CHECKING
    assert export in package.__all__
    assert export not in package.__dict__
    assert export not in package.__dir__()
    assert engine_module not in sys.modules

    resolved = getattr(package, export)
    assert resolved.__name__ == export
    assert sys.modules[engine_module] is importlib.import_module(engine_module)
    assert package.__dict__[export] is resolved
    assert export in package.__dir__()
    assert getattr(package, export) is resolved

    with pytest.raises(AttributeError, match="unknown_export"):
        getattr(package, "unknown_export")


def test_type_checking_exports_do_not_import_engines_at_runtime() -> None:
    code = """
import importlib
import json
import sys

packages = (
    "treemendous.applications.allocation",
    "treemendous.applications.leasing",
    "treemendous.applications.partitioning",
    "treemendous.applications.scheduling",
)
loaded = {}
for package_name in packages:
    package = importlib.import_module(package_name)
    engine_modules = {
        f"{package_name}.{module_name}"
        for module_name, _ in package._EXPORTS.values()
    }
    loaded[package_name] = {
        "type_checking": package.TYPE_CHECKING,
        "engines": sorted(engine_modules.intersection(sys.modules)),
    }
print(json.dumps(loaded, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    try:
        observed: dict[str, dict[str, object]] = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(f"subprocess returned invalid JSON: {exc}: {completed.stdout!r}")
    expected = {
        package_name: {"type_checking": False, "engines": []}
        for package_name, _, _ in FAMILY_EXPORTS
    }
    assert observed == expected
