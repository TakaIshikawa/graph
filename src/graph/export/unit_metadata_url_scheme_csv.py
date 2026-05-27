"""CSV export for URL-like values in unit metadata grouped by key path."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "metadata_key", "scheme", "url"]
_URLISH_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|[a-z][a-z0-9+.-]*:|www\.|/)[^\s]+$", re.IGNORECASE)


def export_unit_metadata_url_scheme_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str]] = []
    for unit in unit_list:
        title = field_value(get(unit, "title") or metadata(unit).get("title"))
        for key, value in _walk(metadata(unit)):
            url = field_value(value)
            if _URLISH_RE.match(url):
                rows.append({"unit_id": unit_id(unit), "title": title, "metadata_key": key, "scheme": _scheme(url), "url": url})
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["metadata_key"]), sort_key(row["url"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _walk(value: Any, prefix: str = "metadata") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        return [item for key in sorted(value, key=sort_key) for item in _walk(value[key], f"{prefix}.{field_value(key)}")]
    if isinstance(value, list | tuple):
        return [item for index, child in enumerate(value) for item in _walk(child, f"{prefix}.{index}")]
    return [(prefix, value)]


def _scheme(url: str) -> str:
    parsed = urlparse(url)
    return parsed.scheme.casefold() if parsed.scheme else "missing"
