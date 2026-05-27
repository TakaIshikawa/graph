"""Summarize duplicate collection titles."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_ID_KEYS = ("id", "collection_id", "source_id")
_TITLE_KEYS = ("title", "name", "label")
_SPACE_RE = re.compile(r"\s+")


def summarize_collection_duplicate_titles(collections: Iterable[Any]) -> dict[str, Any]:
    rows = list(collections)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for index, collection in enumerate(rows):
        title = _title(collection)
        normalized = _normalize_title(title)
        if not normalized:
            continue
        grouped[normalized].append({"collection_id": _collection_id(collection, index), "title": title})

    groups = [
        {
            "normalized_title": title,
            "collection_ids": [item["collection_id"] for item in sorted(items, key=lambda item: sort_key(item["collection_id"]))],
            "titles": [item["title"] for item in sorted(items, key=lambda item: sort_key(item["collection_id"]))],
        }
        for title, items in sorted(grouped.items(), key=lambda item: sort_key(item[0]))
        if len(items) > 1
    ]
    duplicate_collection_count = sum(len(group["collection_ids"]) for group in groups)

    return {
        "collection_count": len(rows),
        "duplicate_title_group_count": len(groups),
        "duplicate_collection_count": duplicate_collection_count,
        "groups": groups,
    }


def _title(collection: Any) -> str:
    meta = metadata(collection)
    for key in _TITLE_KEYS:
        value = _text(get(collection, key)) or _text(meta.get(key))
        if value:
            return value
    return ""


def _collection_id(collection: Any, index: int) -> str:
    meta = metadata(collection)
    for key in _ID_KEYS:
        value = field_value(get(collection, key)) or field_value(meta.get(key))
        if value:
            return value
    return str(index)


def _normalize_title(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip()).casefold()


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value)).strip()
