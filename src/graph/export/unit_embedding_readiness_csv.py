"""CSV export for unit embedding readiness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import (
    field_value,
    get,
    metadata,
    render_csv,
    sort_key,
    unit_id,
    write_csv,
)

_FIELDNAMES = [
    "unit_id",
    "title",
    "content_length",
    "has_language",
    "has_source",
    "has_sensitive_flags",
    "readiness_score",
    "blockers",
]
_LANGUAGE_KEYS = ("language", "lang", "locale")
_SOURCE_KEYS = ("source", "source_id", "source_url", "url", "citation", "source_name")
_SENSITIVE_KEYS = ("sensitive", "sensitive_flags", "pii", "private", "contains_pii")
_DUPLICATE_KEYS = ("duplicate", "is_duplicate", "duplicate_of", "duplicate_content")


def export_unit_embedding_readiness_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write deterministic embedding readiness scores."""
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {
        "path": output_path,
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": bytes_written,
    }


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    content = field_value(get(unit, "content"))
    data = metadata(unit)
    has_language = any(
        field_value(data.get(key)) or field_value(get(unit, key)) for key in _LANGUAGE_KEYS
    )
    has_source = any(
        field_value(data.get(key)) or field_value(get(unit, key)) for key in _SOURCE_KEYS
    )
    has_sensitive = any(
        _truthy(data.get(key)) or _truthy(get(unit, key)) for key in _SENSITIVE_KEYS
    )
    duplicate = any(_truthy(data.get(key)) or _truthy(get(unit, key)) for key in _DUPLICATE_KEYS)
    blockers = []
    if not content:
        blockers.append("empty_content")
    if has_sensitive:
        blockers.append("sensitive_flags")
    if not has_source:
        blockers.append("missing_source")
    if duplicate:
        blockers.append("duplicate_content")
    score = 100
    if not content:
        score -= 50
    if has_sensitive:
        score -= 40
    if not has_source:
        score -= 20
    if duplicate:
        score -= 20
    if not has_language:
        score -= 10
    return {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title")),
        "content_length": len(content),
        "has_language": _flag(has_language),
        "has_source": _flag(has_source),
        "has_sensitive_flags": _flag(has_sensitive),
        "readiness_score": max(0, score),
        "blockers": ";".join(blockers),
    }


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, list | tuple | set):
        return any(_truthy(item) for item in value)
    if isinstance(value, Mapping):
        return any(_truthy(item) for item in value.values())
    return field_value(value).casefold() in {
        "1",
        "true",
        "yes",
        "y",
        "pii",
        "sensitive",
        "duplicate",
    }


def _flag(value: bool) -> str:
    return "true" if value else "false"
