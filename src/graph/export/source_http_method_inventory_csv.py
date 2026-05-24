"""CSV export for source HTTP method hints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, normalized_key, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["http_method", "count", "source_ids", "urls", "source_keys"]
_METHOD_KEYS = {"method", "http_method", "request_method", "fetch_method"}
_NESTED_KEYS = {"request", "fetch", "http", "metadata"}
_URL_KEYS = ("url", "source_url", "canonical_url")


def export_source_http_method_inventory_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write HTTP method counts across sources."""
    source_list = list(sources)
    rows = _rows(source_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "source_ids": set(), "urls": set(), "source_keys": set()})
    for source in sources:
        method, source_key = _method(source)
        bucket = groups[method or "UNKNOWN"]
        bucket["count"] += 1
        if source_id(source):
            bucket["source_ids"].add(source_id(source))
        if _url(source):
            bucket["urls"].add(_url(source))
        bucket["source_keys"].add(source_key or "missing")
    rows: list[dict[str, str | int]] = []
    for method in sorted(groups, key=sort_key):
        bucket = groups[method]
        rows.append(
            {
                "http_method": method,
                "count": bucket["count"],
                "source_ids": "; ".join(sorted(bucket["source_ids"], key=sort_key)),
                "urls": "; ".join(sorted(bucket["urls"], key=sort_key)),
                "source_keys": "; ".join(sorted(bucket["source_keys"], key=sort_key)),
            }
        )
    return rows


def _method(source: Mapping[str, Any] | object) -> tuple[str, str]:
    for key in _METHOD_KEYS:
        text = field_value(get(source, key))
        if text:
            return text.upper(), key
    found = _find_method(metadata(source), "metadata")
    if found[1]:
        return found
    for key in _NESTED_KEYS:
        value = get(source, key)
        if isinstance(value, Mapping):
            found = _find_method(value, key)
            if found[1]:
                return found
    return "UNKNOWN", ""


def _find_method(values: Mapping[str, Any], prefix: str) -> tuple[str, str]:
    for key, value in values.items():
        key_text = field_value(key)
        if normalized_key(key_text) in _METHOD_KEYS and field_value(value):
            return field_value(value).upper(), f"{prefix}.{key_text}"
        if normalized_key(key_text) in _NESTED_KEYS and isinstance(value, Mapping):
            found = _find_method(value, f"{prefix}.{key_text}")
            if found[1]:
                return found
    return "", ""


def _url(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for key in _URL_KEYS:
        text = field_value(get(source, key)) or field_value(data.get(key))
        if text:
            return text
    return ""
