"""CSV export for source language inventory by collection and status."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, normalized_key, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["language", "collection", "status", "count", "source_ids", "source_keys"]
_LANGUAGE_KEYS = {"language", "lang", "locale", "content_language", "source_language", "normalized_language"}
_COLLECTION_KEYS = {"collection", "collections", "collection_id", "collection_name", "folder", "project", "list", "notebook"}
_STATUS_KEYS = {"status", "source_status", "state", "lifecycle_status"}
_UNKNOWN = "unknown"


def export_source_language_inventory_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source language counts grouped by collection and status."""
    source_list = list(sources)
    rows = _inventory_rows(source_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _inventory_rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"count": 0, "source_ids": set(), "source_keys": set()})
    for source in sources:
        languages = _language_values(source)
        if not languages:
            languages = [(_UNKNOWN, "missing")]
        collections = _dimension_values(source, _COLLECTION_KEYS) or [_UNKNOWN]
        statuses = _dimension_values(source, _STATUS_KEYS) or [_UNKNOWN]
        for language, source_key in languages:
            for collection in collections:
                for status in statuses:
                    bucket = buckets[(language, collection, status)]
                    bucket["count"] += 1
                    if source_id(source):
                        bucket["source_ids"].add(source_id(source))
                    bucket["source_keys"].add(source_key)

    rows: list[dict[str, str | int]] = []
    for language, collection, status in sorted(buckets, key=lambda key: (sort_key(key[0]), sort_key(key[1]), sort_key(key[2]))):
        bucket = buckets[(language, collection, status)]
        rows.append(
            {
                "language": language,
                "collection": collection,
                "status": status,
                "count": bucket["count"],
                "source_ids": "; ".join(sorted(bucket["source_ids"], key=sort_key)),
                "source_keys": "; ".join(sorted(bucket["source_keys"], key=sort_key)),
            }
        )
    return rows


def _language_values(source: Mapping[str, Any] | object) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for key in ("language", "lang", "locale", "content_language", "source_language", "normalized_language"):
        text = _normalize_language(get(source, key))
        if text:
            values.append((text, key))
    for raw_key, value in metadata(source).items():
        key = field_value(raw_key)
        if normalized_key(key) in _LANGUAGE_KEYS:
            for item in flatten_values(value):
                language = _normalize_language(item)
                if language:
                    values.append((language, key))
    return values


def _dimension_values(source: Mapping[str, Any] | object, keys: set[str]) -> list[str]:
    values: set[str] = set()
    for key in keys:
        text = field_value(get(source, key))
        if text:
            values.add(text)
    for raw_key, value in metadata(source).items():
        if normalized_key(raw_key) not in keys:
            continue
        values.update(field_value(item) for item in flatten_values(value) if field_value(item))
    return sorted(values, key=sort_key)


def _normalize_language(value: object) -> str:
    text = field_value(value).replace("_", "-").casefold()
    if not text:
        return ""
    text = text.split(",", 1)[0].split(";", 1)[0].strip()
    text = text.split("-", 1)[0]
    return re.sub(r"[^a-z]", "", text) or _UNKNOWN
