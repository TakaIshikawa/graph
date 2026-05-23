"""CSV export for source HTTP cache header inventory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, parse_datetime, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["source_id", "etag_present", "last_modified", "cache_control_present", "expires", "cache_header_bucket"]
_HEADER_KEYS = {"headers", "http_headers", "response_headers", "cache_headers"}
_ALIASES = {
    "etag": ("etag", "e_tag"),
    "last_modified": ("last_modified", "last-modified", "http_last_modified"),
    "cache_control": ("cache_control", "cache-control"),
    "expires": ("expires", "http_expires"),
}


def export_source_cache_header_inventory_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-source HTTP cache header presence rows."""
    source_list = list(sources)
    rows = [_row(source) for source in source_list]
    rows.sort(key=lambda row: sort_key(row["source_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(source: Mapping[str, Any] | object) -> dict[str, str]:
    etag = _header_value(source, "etag")
    last_modified = _date_text(_header_value(source, "last_modified"))
    cache_control = _header_value(source, "cache_control")
    expires = _date_text(_header_value(source, "expires"))
    present = [name for name, value in (("etag", etag), ("last_modified", last_modified), ("cache_control", cache_control), ("expires", expires)) if value]
    return {
        "source_id": source_id(source),
        "etag_present": _bool(etag),
        "last_modified": last_modified,
        "cache_control_present": _bool(cache_control),
        "expires": expires,
        "cache_header_bucket": "+".join(present) if present else "none",
    }


def _header_value(source: Mapping[str, Any] | object, canonical: str) -> str:
    for alias in _ALIASES[canonical]:
        text = field_value(get(source, alias))
        if text:
            return text
    for candidate in (metadata(source), *_header_maps(source)):
        for key, value in candidate.items():
            if normalized_key(key) in {normalized_key(alias) for alias in _ALIASES[canonical]}:
                text = field_value(value)
                if text:
                    return text
    return ""


def _header_maps(source: Mapping[str, Any] | object) -> list[Mapping[str, Any]]:
    maps: list[Mapping[str, Any]] = []
    for key in _HEADER_KEYS:
        value = get(source, key)
        if isinstance(value, Mapping):
            maps.append(value)
    for key, value in metadata(source).items():
        if normalized_key(key) in _HEADER_KEYS and isinstance(value, Mapping):
            maps.append(value)
    return maps


def _date_text(value: object) -> str:
    parsed = parse_datetime(value)
    return parsed.isoformat() if parsed else field_value(value)


def _bool(value: object) -> str:
    return "true" if field_value(value) else "false"
