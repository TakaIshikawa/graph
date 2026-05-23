"""CSV export for source character encoding inventory."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, normalized_key, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["charset", "source_count", "source_ids", "source_keys"]
_CHARSET_KEYS = {"charset", "encoding", "character_encoding", "content_encoding", "content_charset"}
_CONTENT_TYPE_KEYS = {"content_type", "mime_type", "media_type", "http_content_type"}
_HEADER_KEYS = {"headers", "http_headers", "response_headers", "content_headers"}
_UNKNOWN = "unknown"
_CHARSET_RE = re.compile(r"(?:^|;)\s*charset\s*=\s*\"?([^\";]+)", re.IGNORECASE)


def export_source_charset_inventory_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source counts grouped by character encoding hints."""
    source_list = list(sources)
    rows = _inventory_rows(source_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _inventory_rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"source_refs": set(), "source_ids": set(), "source_keys": set()})
    for index, source in enumerate(sources):
        hints = _charset_hints(source) or [(_UNKNOWN, "missing")]
        source_ref = source_id(source) or f"#{index}"
        seen: set[str] = set()
        for charset, source_key in hints:
            bucket_key = charset or _UNKNOWN
            seen.add(bucket_key)
            if source_id(source):
                buckets[bucket_key]["source_ids"].add(source_id(source))
            buckets[bucket_key]["source_keys"].add(source_key)
        for bucket_key in seen:
            buckets[bucket_key]["source_refs"].add(source_ref)

    rows: list[dict[str, str | int]] = []
    for charset, bucket in buckets.items():
        rows.append(
            {
                "charset": charset,
                "source_count": len(bucket["source_refs"]),
                "source_ids": "; ".join(sorted(bucket["source_ids"], key=sort_key)),
                "source_keys": "; ".join(sorted(bucket["source_keys"], key=sort_key)),
            }
        )
    return sorted(rows, key=lambda row: sort_key(row["charset"]))


def _charset_hints(source: Mapping[str, Any] | object) -> list[tuple[str, str]]:
    hints: list[tuple[str, str]] = []
    for key in _CHARSET_KEYS | _CONTENT_TYPE_KEYS:
        hints.extend(_values_for_key(source, key))
    for raw_key, value in metadata(source).items():
        key = field_value(raw_key)
        norm = normalized_key(key)
        if norm in _CHARSET_KEYS:
            hints.extend((_normalize_charset(item), key) for item in flatten_values(value) if _normalize_charset(item))
        elif norm in _CONTENT_TYPE_KEYS:
            hints.extend((_charset_from_content_type(item), key) for item in flatten_values(value) if _charset_from_content_type(item))
        elif norm in _HEADER_KEYS and isinstance(value, Mapping):
            hints.extend(_header_hints(value, key))
    for key in _HEADER_KEYS:
        value = get(source, key)
        if isinstance(value, Mapping):
            hints.extend(_header_hints(value, key))
    return hints


def _values_for_key(source: Mapping[str, Any] | object, key: str) -> list[tuple[str, str]]:
    value = get(source, key)
    if normalized_key(key) in _CONTENT_TYPE_KEYS:
        return [(_charset_from_content_type(item), key) for item in flatten_values(value) if _charset_from_content_type(item)]
    return [(_normalize_charset(item), key) for item in flatten_values(value) if _normalize_charset(item)]


def _header_hints(headers: Mapping[str, Any], source_key: str) -> list[tuple[str, str]]:
    hints: list[tuple[str, str]] = []
    for header_key, value in headers.items():
        norm = normalized_key(header_key)
        if norm in _CHARSET_KEYS:
            hints.extend((_normalize_charset(item), f"{source_key}.{field_value(header_key)}") for item in flatten_values(value) if _normalize_charset(item))
        elif norm in _CONTENT_TYPE_KEYS:
            hints.extend((_charset_from_content_type(item), f"{source_key}.{field_value(header_key)}") for item in flatten_values(value) if _charset_from_content_type(item))
    return hints


def _charset_from_content_type(value: object) -> str:
    match = _CHARSET_RE.search(field_value(value))
    return _normalize_charset(match.group(1)) if match else ""


def _normalize_charset(value: object) -> str:
    return field_value(value).strip("\"'").casefold().replace("_", "-")
