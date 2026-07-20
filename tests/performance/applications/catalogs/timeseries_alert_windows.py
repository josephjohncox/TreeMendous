"""Smoke benchmark for alert/suppression evaluation."""

from time import perf_counter

from treemendous.applications.catalogs.timeseries_alert_windows import AlertCatalog


def run_smoke(iterations: int = 500) -> float:
    catalog = AlertCatalog()
    for index in range(200):
        catalog.add(
            f"a{index}",
            index * 10,
            index * 10 + 100,
            series="cpu",
            kind="alert",
            priority=index % 10,
            label="cpu",
        )
    for index in range(20):
        catalog.add(
            f"s{index}",
            index * 100,
            index * 100 + 20,
            series="cpu",
            kind="suppression",
            priority=5,
            label="maintenance",
        )
    started = perf_counter()
    for index in range(iterations):
        catalog.active_at("cpu", index % 2_000)
    return perf_counter() - started


if __name__ == "__main__":
    print(run_smoke())
