"""CSV export for source identifier coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from graph.export._report_csv import (
    field_value,
    flatten_values,
    get,
    metadata,
    render_csv,
    sort_key,
    source_id,
    write_csv,
)

_FIELDNAMES = [
    "source_id",
    "source_name",
    "identifier_count",
    "identifiers_present",
    "identifiers_missing",
    "coverage_score",
]
_DEFAULT_KEYS = ("url", "external_id", "account_id", "feed_url", "doi", "isbn", "handle")


def export_source_identifier_coverage_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    *,
    identifier_keys: Sequence[str] = _DEFAULT_KEYS,
) -> str | dict[str, Any]:
    """Return or write stable identifier coverage for sources."""
    source_list = list(sources)
    rows = sorted(
        (_row(source, tuple(identifier_keys)) for source in source_list),
        key=lambda row: sort_key(row["source_id"]),
    )
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


def _row(source: Mapping[str, Any] | object, keys: tuple[str, ...]) -> dict[str, str | int]:
    present = [
        key for key in keys if _present(get(source, key)) or _present(metadata(source).get(key))
    ]
    missing = [key for key in keys if key not in present]
    score = len(present) / len(keys) if keys else 0
    return {
        "source_id": source_id(source),
        "source_name": field_value(
            get(source, "name") or get(source, "title") or metadata(source).get("name")
        ),
        "identifier_count": len(present),
        "identifiers_present": ";".join(present),
        "identifiers_missing": ";".join(missing),
        "coverage_score": f"{score:.2f}",
    }


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(_present(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(field_value(item) for item in flatten_values(value))
    return bool(field_value(value))
