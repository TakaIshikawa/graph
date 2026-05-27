"""Summarize source payload sizes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

OVERSIZED_PAYLOAD_BYTES = 1_000_000

_SOURCE_KEYS = ("source_project", "source")
_SIZE_KEYS = ("payload_bytes", "content_length", "size_bytes", "file_size")
_ID_KEYS = ("id", "unit_id")


def source_payload_size_summary(units: Iterable[Any]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for unit in units:
        metadata = _metadata(unit)
        source = _string(_first(unit, metadata, _SOURCE_KEYS)) or "unknown"
        size = _size(unit, metadata)
        group = groups.setdefault(source, {"source": source, "sizes": [], "unit_count": 0, "oversized_count": 0, "sample_unit_ids": []})
        group["unit_count"] += 1
        group["sizes"].append(size)
        if size > OVERSIZED_PAYLOAD_BYTES:
            group["oversized_count"] += 1
        unit_id = _string(_first(unit, metadata, _ID_KEYS))
        if unit_id and len(group["sample_unit_ids"]) < 3:
            group["sample_unit_ids"].append(unit_id)

    rows = []
    for group in groups.values():
        sizes = group["sizes"]
        rows.append(
            {
                "source": group["source"],
                "unit_count": group["unit_count"],
                "min_bytes": min(sizes),
                "max_bytes": max(sizes),
                "average_bytes": round(sum(sizes) / len(sizes), 2),
                "total_bytes": sum(sizes),
                "oversized_count": group["oversized_count"],
                "sample_unit_ids": group["sample_unit_ids"],
            }
        )
    return sorted(rows, key=lambda row: (-row["unit_count"], row["source"]))


def _size(unit: Any, metadata: Mapping[str, Any]) -> int:
    explicit = _first(unit, metadata, _SIZE_KEYS)
    try:
        size = int(explicit)
        return size if size > 0 else 0
    except (TypeError, ValueError):
        pass
    content = _get(unit, "content") or metadata.get("content")
    if content in (None, ""):
        return 0
    if isinstance(content, bytes):
        return len(content)
    return len(str(content).encode("utf-8"))


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _first(item: Any, metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _get(item, key)
        if value not in (None, ""):
            return value
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
