"""Summarize alt text quality for markdown image references."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")


def summarize_unit_image_alt_text(units: Iterable[Any]) -> dict[str, Any]:
    total_units = total_images = present_alt_text_count = empty_alt_text_count = 0
    missing_alt_text_count = 0
    present_units: set[str] = set()
    empty_units: set[str] = set()
    missing_units: set[str] = set()

    for index, unit in enumerate(units):
        total_units += 1
        identifier = unit_id(unit) or str(index)
        content = _content(unit)
        for match in _IMAGE_RE.finditer(content):
            total_images += 1
            raw_alt = match.group(1)
            if raw_alt is None:
                missing_alt_text_count += 1
                missing_units.add(identifier)
            elif raw_alt.strip():
                present_alt_text_count += 1
                present_units.add(identifier)
            else:
                empty_alt_text_count += 1
                empty_units.add(identifier)

    return {
        "total_units": total_units,
        "total_images": total_images,
        "present_alt_text_count": present_alt_text_count,
        "empty_alt_text_count": empty_alt_text_count,
        "missing_alt_text_count": missing_alt_text_count,
        "units_with_present_alt_text": sorted(present_units, key=sort_key),
        "units_with_empty_alt_text": sorted(empty_units, key=sort_key),
        "units_with_missing_alt_text": sorted(missing_units, key=sort_key),
    }


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)
