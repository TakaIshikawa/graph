"""CSV export for unit attachment size hints."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "attachment_count", "total_bytes", "largest_bytes", "size_bucket", "source_keys"]
_DIRECT_SIZE_KEYS = {"size", "file_size", "filesize", "bytes", "byte_size", "content_length", "attachment_size", "attachment_bytes"}
_ATTACHMENT_KEYS = {"attachment", "attachments", "file", "files"}
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(b|bytes?|kb|kib|mb|mib|gb|gib)?\s*$", re.IGNORECASE)


def export_unit_attachment_size_inventory_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write attachment size totals per unit."""
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    sizes = _sizes(unit)
    total = sum(size for size, _key in sizes)
    largest = max((size for size, _key in sizes), default=0)
    return {
        "unit_id": unit_id(unit),
        "attachment_count": len(sizes),
        "total_bytes": total,
        "largest_bytes": largest,
        "size_bucket": _bucket(total, len(sizes)),
        "source_keys": "; ".join(sorted({key for _size, key in sizes}, key=sort_key)),
    }


def _sizes(unit: Mapping[str, Any] | object) -> list[tuple[int, str]]:
    values: list[tuple[int, str]] = []
    for key in _DIRECT_SIZE_KEYS:
        size = _parse_size(get(unit, key))
        if size is not None:
            values.append((size, key))
    for key, value in metadata(unit).items():
        key_text = field_value(key)
        normalized = key_text.casefold().replace("-", "_").replace(" ", "_")
        if normalized in _DIRECT_SIZE_KEYS:
            for item in flatten_values(value):
                size = _parse_size(item)
                if size is not None:
                    values.append((size, key_text))
        if normalized in _ATTACHMENT_KEYS:
            values.extend(_attachment_sizes(value, key_text))
    return values


def _attachment_sizes(value: object, source_key: str) -> list[tuple[int, str]]:
    sizes: list[tuple[int, str]] = []
    items = value if isinstance(value, list | tuple | set) else [value]
    for item in items:
        if isinstance(item, Mapping):
            for key, raw in item.items():
                if field_value(key).casefold().replace("-", "_").replace(" ", "_") in _DIRECT_SIZE_KEYS:
                    size = _parse_size(raw)
                    if size is not None:
                        sizes.append((size, f"{source_key}.{field_value(key)}"))
        else:
            size = _parse_size(item)
            if size is not None:
                sizes.append((size, source_key))
    return sizes


def _parse_size(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return int(value) if value >= 0 else None
    match = _SIZE_RE.match(field_value(value).replace(",", ""))
    if not match:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "b").casefold()
    multiplier = {"b": 1, "byte": 1, "bytes": 1, "kb": 1000, "kib": 1024, "mb": 1000**2, "mib": 1024**2, "gb": 1000**3, "gib": 1024**3}[unit]
    return int(amount * multiplier)


def _bucket(total: int, count: int) -> str:
    if count == 0:
        return "missing"
    if total < 1024 * 1024:
        return "small"
    if total < 100 * 1024 * 1024:
        return "medium"
    return "large"
