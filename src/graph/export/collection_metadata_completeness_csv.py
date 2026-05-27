"""CSV export for collection metadata completeness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, write_csv

_FIELDNAMES = ["collection_id", "title", "required_key_count", "present_key_count", "missing_keys", "empty_keys", "completeness_ratio"]
DEFAULT_REQUIRED_KEYS = ("title", "description", "source", "updated_at")


def export_collection_metadata_completeness_csv(collections: Iterable[Mapping[str, Any] | object], path: str | Path | None = None, *, required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS) -> str | dict[str, Any]:
    collection_list = list(collections)
    rows = [_row(collection, required_keys) for collection in collection_list]
    rows.sort(key=lambda row: (sort_key(row["collection_id"]), sort_key(row["title"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "collection_count": len(collection_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(collection: Mapping[str, Any] | object, required_keys: tuple[str, ...]) -> dict[str, str | int]:
    empty = [key for key in required_keys if not _present(_value(collection, key))]
    present = len(required_keys) - len(empty)
    return {
        "collection_id": field_value(get(collection, "id") or get(collection, "collection_id") or metadata(collection).get("id")),
        "title": field_value(get(collection, "title") or get(collection, "name") or metadata(collection).get("title") or metadata(collection).get("name")),
        "required_key_count": len(required_keys),
        "present_key_count": present,
        "missing_keys": "; ".join(key for key in required_keys if key not in _all_keys(collection)),
        "empty_keys": "; ".join(empty),
        "completeness_ratio": f"{(present / len(required_keys)):.2f}" if required_keys else "0.00",
    }


def _value(collection: Mapping[str, Any] | object, key: str) -> object:
    value = get(collection, key)
    return value if value not in (None, "") else metadata(collection).get(key)


def _all_keys(collection: Mapping[str, Any] | object) -> set[str]:
    keys = set(collection.keys()) if isinstance(collection, Mapping) else set()
    keys.update(metadata(collection).keys())
    return keys


def _present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping | list | tuple | set):
        return bool(value)
    return True
