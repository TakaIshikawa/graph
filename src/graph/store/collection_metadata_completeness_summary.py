"""Metadata completeness summary for store collections."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_REQUIRED_KEYS = ("title", "description", "source", "updated_at")


def summarize_collection_metadata_completeness(
    collections: Iterable[Any], *, required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS
) -> dict[str, Any]:
    items = list(collections)
    missing = []
    coverage = {}
    present_total = 0
    possible_total = len(items) * len(required_keys)
    for key in required_keys:
        present = sum(1 for item in items if _present(_value(item, key)))
        present_total += present
        coverage[key] = {
            "present_count": present,
            "missing_count": len(items) - present,
            "coverage_ratio": f"{(present / len(items)):.2f}" if items else "0.00",
        }
    for index, item in enumerate(items):
        missing_keys = [key for key in required_keys if not _present(_value(item, key))]
        if missing_keys:
            missing.append({"collection_id": _collection_id(item, index), "missing_keys": missing_keys})
    return {
        "total_collections": len(items),
        "required_keys": list(required_keys),
        "required_key_coverage": coverage,
        "missing_by_collection": sorted(missing, key=lambda row: _sort_key(row["collection_id"])),
        "overall_coverage_ratio": f"{(present_total / possible_total):.2f}" if possible_total else "0.00",
    }


def _value(item: Any, key: str) -> Any:
    value = _get(item, key)
    return value if value not in (None, "") else _metadata(item).get(key)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list | tuple | set):
        return bool(value)
    return True


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _collection_id(item: Any, index: int) -> str:
    return _text(_get(item, "id") or _metadata(item).get("id")) or str(index)


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
