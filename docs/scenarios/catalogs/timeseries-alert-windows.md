# Time-series alert windows

## Model

`AlertCatalog` stores alert and suppression windows as separate identities. Each record retains window ID, series, kind, integer priority, label, active interval, and insertion order. This prevents a union of time ranges from erasing whether activity came from a firing rule or maintenance policy.

At a timestamp, `active_at` isolates one series and finds every active alert and suppression. The highest suppression priority suppresses alerts with equal or lower priority; strictly higher-priority alerts still fire. Returned firing alerts, suppressions, and suppressed alerts are each ordered by descending priority and then insertion order. `active_windows` exposes all policies intersecting a time range using the same deterministic order.

## Mutation and validation

Kind is exactly `alert` or `suppression`; IDs, series names, and labels are nonempty. Priorities are integers and windows are nonempty half-open spans. `update` can alter timing, kind, priority, or label while retaining handle identity. `remove` affects one window. `snapshot` is immutable and insertion ordered.

## Example

```python
catalog.add("cpu-high", 0, 100, series="host.cpu", kind="alert",
            priority=10, label="CPU high")
catalog.add("deploy", 20, 40, series="host.cpu", kind="suppression",
            priority=5, label="deployment")
```

The documented rule is priority based; label matching and cross-series suppression are intentionally not implied.
