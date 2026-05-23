"""CSV export for unit locale metadata coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "locale", "language", "country", "region", "timezone", "currency", "locale_bucket", "missing_locale_fields"]
_FIELDS = ("locale", "language", "country", "region", "timezone", "currency")
_ALIASES = {"language": ("language", "lang"), "timezone": ("timezone", "time_zone", "tz")}
_UNKNOWN = "unknown"


def export_unit_locale_coverage_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one row per unit with locale signal coverage."""
    unit_list = list(units)
    rows = [_row(unit) for unit in unit_list]
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str]:
    values = {field: _field(unit, field) for field in _FIELDS}
    missing = [field for field, value in values.items() if not value]
    return {
        "unit_id": unit_id(unit),
        **values,
        "locale_bucket": _locale_bucket(values),
        "missing_locale_fields": "; ".join(missing),
    }


def _field(unit: Mapping[str, Any] | object, field: str) -> str:
    aliases = _ALIASES.get(field, (field,))
    for alias in aliases:
        text = field_value(get(unit, alias))
        if text:
            return text
    alias_keys = {normalized_key(alias) for alias in aliases}
    for key, value in metadata(unit).items():
        if normalized_key(key) in alias_keys and field_value(value):
            return field_value(value)
    return ""


def _locale_bucket(values: dict[str, str]) -> str:
    locale = values["locale"].replace("_", "-").casefold()
    if locale:
        return locale
    parts = [values["language"].casefold(), values["country"].casefold()]
    bucket = "-".join(part for part in parts if part)
    return bucket or _UNKNOWN
