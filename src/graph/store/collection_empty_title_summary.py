"""Summarize missing and blank collection titles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_ID_KEYS = ("id", "collection_id", "source_id")
_TITLE_KEYS = ("title", "name", "label")


def summarize_collection_empty_titles(collections: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    rows = list(collections)
    missing = blank = present = 0
    samples: list[str] = []
    for index, collection in enumerate(rows):
        meta = metadata(collection)
        status = _title_status(collection, meta)
        if status == "missing":
            missing += 1
        elif status == "blank":
            blank += 1
        else:
            present += 1
            continue
        if len(samples) < max(0, sample_limit):
            samples.append(_first(collection, meta, _ID_KEYS) or str(index))
    total = len(rows)
    return {
        "total_collections": total,
        "missing_title_count": missing,
        "blank_title_count": blank,
        "present_title_count": present,
        "completeness_ratio": round(present / total, 4) if total else 0,
        "sample_collection_ids": sorted(samples, key=sort_key)[: max(0, sample_limit)],
    }


def _title_status(item: Any, meta: Mapping[str, Any]) -> str:
    saw_field = False
    for key in _TITLE_KEYS:
        for value in (get(item, key), meta.get(key)):
            if value is None:
                continue
            saw_field = True
            if field_value(value):
                return "present"
    return "blank" if saw_field else "missing"


def _first(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""
