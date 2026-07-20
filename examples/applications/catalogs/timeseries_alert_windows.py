"""Evaluate alerts against an active suppression priority."""

from treemendous.applications.catalogs.timeseries_alert_windows import AlertCatalog


def main() -> None:
    catalog = AlertCatalog()
    catalog.add(
        "cpu", 0, 100, series="host.cpu", kind="alert", priority=10, label="high CPU"
    )
    catalog.add(
        "deploy",
        20,
        40,
        series="host.cpu",
        kind="suppression",
        priority=5,
        label="deployment",
    )
    print(catalog.active_at("host.cpu", 30).alerts)


if __name__ == "__main__":
    main()
