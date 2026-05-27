"""CSV export for URL values found in unit metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "source", "metadata_key_path", "url", "hostname", "scheme"]
_URL_RE = re.compile(r"https?://[^\s<>\]\"']+|www\.[^\s<>\]\"']+", re.IGNORECASE)


def export_units_to_metadata_url_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str]] = []
    for unit in unit_list:
        source = field_value(get(unit, "source") or metadata(unit).get("source"))
        for key_path, value in _walk(metadata(unit)):
            if not isinstance(value, str):
                continue
            for url in _urls(value):
                parsed = _parse_url(url)
                rows.append(
                    {
                        "unit_id": unit_id(unit),
                        "source": source,
                        "metadata_key_path": key_path,
                        "url": url,
                        "hostname": parsed.hostname.casefold() if parsed.hostname else "",
                        "scheme": parsed.scheme.casefold(),
                    }
                )
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["metadata_key_path"]), sort_key(row["url"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _walk(value: Any, prefix: str = "metadata") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        return [
            item
            for key in sorted(value, key=sort_key)
            for item in _walk(value[key], f"{prefix}.{field_value(key)}")
        ]
    if isinstance(value, list | tuple):
        return [item for index, child in enumerate(value) for item in _walk(child, f"{prefix}[{index}]")]
    return [(prefix, value)]


def _urls(value: str) -> list[str]:
    return [match.group(0).rstrip(".,;:!?)]}") for match in _URL_RE.finditer(value)]


def _parse_url(url: str):
    if url.casefold().startswith("www."):
        return urlparse(f"https://{url}")
    return urlparse(url)
