"""CSV export for DOI inventory across units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import (
    field_value,
    flatten_values,
    get,
    metadata,
    render_csv,
    sort_key,
    unit_id,
    write_csv,
)

_FIELDNAMES = ["unit_id", "title", "doi", "doi_source", "normalized_doi", "has_doi"]
_DOI_KEYS = {"doi", "digital_object_identifier", "identifier_doi"}
_URL_KEYS = ("url", "source_url", "external_url", "canonical_url", "link")
_DOI_RE = re.compile(
    r"(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.IGNORECASE
)


def export_unit_doi_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write DOI coverage rows for units."""
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


def _row(unit: Mapping[str, Any] | object) -> dict[str, str]:
    doi, source = _doi(unit)
    normalized = _normalize_doi(doi)
    return {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title")),
        "doi": doi,
        "doi_source": source,
        "normalized_doi": normalized,
        "has_doi": "true" if normalized else "false",
    }


def _doi(unit: Mapping[str, Any] | object) -> tuple[str, str]:
    data = metadata(unit)
    for key, value in data.items():
        if field_value(key).casefold().replace("-", "_") in _DOI_KEYS:
            found = _extract(value)
            if found:
                return found, f"metadata.{key}"
    for key in _URL_KEYS:
        found = _extract(get(unit, key) or data.get(key))
        if found:
            return found, key
    found = _extract(get(unit, "content"))
    return (found, "content") if found else ("", "")


def _extract(value: object) -> str:
    for item in flatten_values(value):
        match = _DOI_RE.search(field_value(item).rstrip(".,;)"))
        if match:
            return match.group(1).rstrip(".,;)")
    return ""


def _normalize_doi(value: str) -> str:
    match = _DOI_RE.search(field_value(value))
    return match.group(1).rstrip(".,;)").casefold() if match else ""
