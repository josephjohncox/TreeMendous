"""Naive versioned diagnostic range oracle."""

from __future__ import annotations

DiagnosticRow = tuple[str, str, int, int, int, int]


def query(
    rows: list[DiagnosticRow],
    file: str,
    version: int,
    start: int,
    end: int,
    minimum_severity: int,
) -> tuple[str, ...]:
    """Scan and severity-order bounded diagnostic rows."""
    matches = [
        row
        for row in rows
        if row[1] == file
        and row[2] == version
        and row[3] >= minimum_severity
        and row[4] < end
        and start < row[5]
    ]
    matches.sort(key=lambda row: -row[3])
    return tuple(row[0] for row in matches)


def remap_edit(
    rows: list[DiagnosticRow],
    file: str,
    version: int,
    edit_start: int,
    edit_end: int,
    replacement_length: int,
    *,
    new_version: int,
) -> tuple[DiagnosticRow, ...]:
    """Remap rows using a direct endpoint reference model.

    At an insertion boundary, starts attach to the following text and ends
    attach to the preceding text. An enclosing range therefore grows to cover
    inserted text, while a range touching only one side does not.
    """
    insertion = edit_start == edit_end
    delta = replacement_length - (edit_end - edit_start)

    def map_start(offset: int) -> int:
        if insertion:
            return offset if offset < edit_start else offset + replacement_length
        if offset <= edit_start:
            return offset
        if offset >= edit_end:
            return offset + delta
        return edit_start + min(offset - edit_start, replacement_length)

    def map_end(offset: int) -> int:
        if insertion:
            return offset if offset <= edit_start else offset + replacement_length
        if offset <= edit_start:
            return offset
        if offset >= edit_end:
            return offset + delta
        return edit_start + min(offset - edit_start, replacement_length)

    result: list[DiagnosticRow] = []
    for diagnostic_id, row_file, row_version, severity, start, end in rows:
        if row_file != file or row_version != version:
            result.append((diagnostic_id, row_file, row_version, severity, start, end))
            continue
        mapped_start = map_start(start)
        mapped_end = map_end(end)
        if mapped_start < mapped_end:
            result.append(
                (
                    diagnostic_id,
                    row_file,
                    new_version,
                    severity,
                    mapped_start,
                    mapped_end,
                )
            )
    return tuple(result)
