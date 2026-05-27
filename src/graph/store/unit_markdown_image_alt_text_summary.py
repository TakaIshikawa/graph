"""Summarize markdown image alt text coverage."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]*)\)")


def summarize_unit_markdown_image_alt_text(units: Iterable[Any]) -> dict[str, Any]:
    total_units = units_with_images = image_count = missing_alt_count = empty_alt_count = present_alt_count = 0
    unit_rows: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        total_units += 1
        content = str(get(unit, "content") or metadata(unit).get("content") or "")
        matches = list(_IMAGE_RE.finditer(content))
        if matches:
            units_with_images += 1
        row = {"unit_id": unit_id(unit) or str(index), "image_count": len(matches), "missing_alt_count": 0, "empty_alt_count": 0, "present_alt_count": 0}
        for match in matches:
            image_count += 1
            alt = match.group(1)
            if alt is None:
                missing_alt_count += 1
                row["missing_alt_count"] += 1
            elif not alt.strip():
                empty_alt_count += 1
                row["empty_alt_count"] += 1
            else:
                present_alt_count += 1
                row["present_alt_count"] += 1
        if matches:
            unit_rows.append(row)
    return {"total_units": total_units, "units_with_images": units_with_images, "image_count": image_count, "missing_alt_count": missing_alt_count, "empty_alt_count": empty_alt_count, "present_alt_count": present_alt_count, "units": sorted(unit_rows, key=lambda row: sort_key(row["unit_id"]))}
