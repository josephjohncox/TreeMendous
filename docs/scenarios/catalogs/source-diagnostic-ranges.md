# Source diagnostic ranges

## Model

`SourceDiagnosticCatalog` preserves every diagnostic identity together with file, source version, severity, message, and half-open byte range. Equal ranges are not unioned. `diagnostics` first applies file and version isolation, then a minimum severity threshold, and returns severity-descending results with insertion order as the deterministic tie breaker.

`remap_edit` advances every diagnostic for one file/version through replacement of `[edit_start, edit_end)`. Offsets before the edit remain fixed, offsets after it shift by the replacement delta, and anchors inside replacement text are clipped to the replacement length. A diagnostic collapsed to an empty range by deletion is removed explicitly. Other files and versions remain untouched.

## Mutation and validation

Versions are nonnegative integers and a remap target must be newer. Severity is one of `INFO`, `WARNING`, `ERROR`, or `FATAL`. File, ID, and message are nonempty. `update` keeps identity while changing range, version, severity, or message; `remove` affects one diagnostic. `snapshot` is an immutable, insertion-ordered state view.

## Example

```python
handle = catalog.add("E101", 12, 16, file="main.py", version=3,
                     severity=Severity.ERROR, message="unknown name")
catalog.remap_edit("main.py", 3, 0, 4, 10, new_version=4)
```

This remapping policy is deterministic and documented; it does not attempt semantic AST relocation.
