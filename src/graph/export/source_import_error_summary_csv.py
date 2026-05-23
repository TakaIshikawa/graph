"""CSV export for source import error summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import (
    field_value,
    get,
    metadata,
    parse_datetime,
    render_csv,
    sort_key,
    source_id,
    write_csv,
)

_FIELDNAMES = [
    "source_id",
    "source_name",
    "error_count",
    "warning_count",
    "last_error_at",
    "last_error_message",
]


def export_source_import_error_summary_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source-level import error counts and latest messages."""
    source_list = list(sources)
    rows = [_row(source) for source in source_list]
    rows.sort(key=lambda row: (-int(row["error_count"]), sort_key(row["source_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {
        "path": output_path,
        "source_count": len(source_list),
        "rows_exported": len(rows),
        "bytes_written": bytes_written,
    }


def _row(source: Mapping[str, Any] | object) -> dict[str, str | int]:
    data = metadata(source)
    errors = _items(get(source, "errors")) + _items(
        data.get("errors") or data.get("import_errors") or data.get("error_messages")
    )
    warnings = _items(get(source, "warnings")) + _items(
        data.get("warnings") or data.get("import_warnings") or data.get("warning_messages")
    )
    error_count = _count(get(source, "error_count"), data.get("error_count"), len(errors))
    warning_count = _count(get(source, "warning_count"), data.get("warning_count"), len(warnings))
    last_error = _last_error(errors)
    return {
        "source_id": source_id(source),
        "source_name": field_value(get(source, "name") or get(source, "title") or data.get("name")),
        "error_count": error_count,
        "warning_count": warning_count,
        "last_error_at": _date_text(
            get(source, "last_error_at")
            or get(source, "error_at")
            or data.get("last_error_at")
            or data.get("error_at")
            or last_error.get("at")
        ),
        "last_error_message": field_value(
            get(source, "last_error_message")
            or get(source, "error_message")
            or data.get("last_error_message")
            or data.get("error_message")
            or last_error.get("message")
        ),
    }


def _count(*values: object) -> int:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            return max(0, int(value))
        text = field_value(value)
        if text.isdigit():
            return int(text)
    return 0


def _items(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list | tuple | set):
        return list(value)
    text = field_value(value)
    return [text] if text else []


def _last_error(errors: list[object]) -> Mapping[str, Any]:
    mappings = [error for error in errors if isinstance(error, Mapping)]
    if not mappings:
        return {"message": errors[-1]} if errors else {}
    return max(mappings, key=_error_sort_key)


def _error_sort_key(error: Mapping[str, Any]) -> tuple[int, str]:
    parsed = parse_datetime(error.get("at") or error.get("error_at") or error.get("timestamp"))
    return (1, parsed.isoformat()) if parsed else (0, "")


def _date_text(value: object) -> str:
    parsed = parse_datetime(value)
    return parsed.isoformat() if parsed else field_value(value)
