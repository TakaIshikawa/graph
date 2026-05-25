"""Summarize store unit attachment types and sizes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import PurePath
from typing import Any

_ATTACHMENT_KEYS = ("attachments", "assets", "files")
_TYPE_KEYS = ("mime_type", "content_type", "type")
_SIZE_KEYS = ("size", "size_bytes", "bytes", "content_length")
_NAME_KEYS = ("filename", "name", "path", "url")
_UNIT_ID_KEYS = ("id", "unit_id")


def summarize_unit_attachment_types(units: Iterable[Any]) -> dict[str, Any]:
    """Aggregate attachment counts and byte totals by MIME/type or extension."""

    total_units = total_attachments = units_missing_metadata = 0
    grouped: dict[str, dict[str, Any]] = defaultdict(_empty_group)

    for unit in units:
        total_units += 1
        unit_id = _unit_id(unit)
        attachments = _attachments(unit)
        unit_missing_metadata = False
        for attachment in attachments:
            total_attachments += 1
            attachment_type = _attachment_type(attachment)
            size = _size(attachment)
            if attachment_type is None or size is None:
                unit_missing_metadata = True

            group_key = attachment_type or "unknown"
            group = grouped[group_key]
            group["type"] = group_key
            group["_unit_ids"].add(unit_id)
            group["attachment_count"] += 1
            if size is not None:
                group["total_bytes"] += size
                group["largest_bytes"] = max(group["largest_bytes"], size)

        if attachments and unit_missing_metadata:
            units_missing_metadata += 1

    rows = []
    for group in grouped.values():
        rows.append(
            {
                "type": group["type"],
                "unit_count": len(group["_unit_ids"]),
                "attachment_count": group["attachment_count"],
                "total_bytes": group["total_bytes"],
                "largest_bytes": group["largest_bytes"],
            }
        )

    return {
        "total_units": total_units,
        "total_attachments": total_attachments,
        "units_missing_attachment_metadata": units_missing_metadata,
        "attachment_types": sorted(rows, key=lambda item: (-item["attachment_count"], item["type"])),
    }


def _empty_group() -> dict[str, Any]:
    return {
        "type": "",
        "_unit_ids": set(),
        "attachment_count": 0,
        "total_bytes": 0,
        "largest_bytes": 0,
    }


def _attachments(unit: Any) -> list[Mapping[str, Any]]:
    metadata = _metadata(unit)
    for key in _ATTACHMENT_KEYS:
        value = _get(unit, key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
        value = metadata.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _attachment_type(attachment: Mapping[str, Any]) -> str | None:
    value = _first(attachment, _TYPE_KEYS)
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    extension = _extension(attachment)
    return extension.lower() if extension else None


def _extension(attachment: Mapping[str, Any]) -> str | None:
    value = attachment.get("extension")
    if isinstance(value, str) and value.strip():
        return value.strip().lower().lstrip(".")
    value = _first(attachment, _NAME_KEYS)
    if not isinstance(value, str) or not value.strip():
        return None
    suffix = PurePath(value.split("?", 1)[0]).suffix
    return suffix.lstrip(".") if suffix else None


def _size(attachment: Mapping[str, Any]) -> int | None:
    value = _first(attachment, _SIZE_KEYS)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _metadata(item: Any) -> Mapping[str, Any]:
    value = _get(item, "metadata")
    return value if isinstance(value, Mapping) else {}


def _unit_id(item: Any) -> str:
    for key in _UNIT_ID_KEYS:
        value = _get(item, key)
        if value not in (None, ""):
            return str(value)
    return ""


def _get(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None
