"""Summarize Markdown image title-attribute coverage."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_IMAGE_RE = re.compile(r"!\[([^\]\n]*)]\(([^)\n]*)\)")
_TITLE_RE = re.compile(r"""^\s*\S+(?:\s+(?:"([^"]*)"|'([^']*)'))?\s*$""")


def summarize_unit_markdown_image_title_attributes(units: Iterable[Any]) -> dict[str, Any]:
    total = image_count = with_title = without_title = 0
    rows: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_images = unit_missing = 0
        for target in _images(str(get(unit, "content") or "")):
            unit_images += 1
            if _has_title(target):
                with_title += 1
            else:
                without_title += 1
                unit_missing += 1
        if unit_images:
            image_count += unit_images
            rows.append({"unit_id": uid, "image_count": unit_images, "missing_title_count": unit_missing})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {
        "total_units": total,
        "image_count": image_count,
        "images_with_title_count": with_title,
        "images_without_title_count": without_title,
        "units_with_images": len(rows),
        "units_missing_titles": sum(1 for row in rows if row["missing_title_count"]),
        "units": rows,
    }


def _images(content: str) -> list[str]:
    rows: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        rows.extend(match.group(2) for match in _IMAGE_RE.finditer(line))
    return rows


def _has_title(target: str) -> bool:
    match = _TITLE_RE.match(target)
    return bool(match and field_value(next((group for group in match.groups() if group is not None), "")))
