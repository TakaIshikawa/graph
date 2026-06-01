"""Summarize Markdown links and images with empty labels."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_EMPTY_LABEL_RE = re.compile(r"(!?)\[\]\(([^)\n]*)\)")


def summarize_unit_markdown_empty_link_texts(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize Markdown inline links/images whose bracket text is empty."""
    limit = max(0, sample_limit)
    total = links = images = 0
    affected: set[str] = set()
    examples: list[dict[str, str | int | bool]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, target, is_image in _empty_labels(str(get(unit, "content") or "")):
            affected.add(uid)
            if is_image:
                images += 1
            else:
                links += 1
            examples.append({"unit_id": uid, "line": line_number, "target": target, "is_image": is_image})
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["target"]), bool(row["is_image"])))
    return {
        "total_units": total,
        "links_with_empty_text": links,
        "images_with_empty_alt": images,
        "affected_units": len(affected),
        "examples": examples[:limit],
    }


def _empty_labels(content: str) -> list[tuple[int, str, bool]]:
    rows: list[tuple[int, str, bool]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _EMPTY_LABEL_RE.finditer(line):
            rows.append((line_number, field_value(match.group(2)), bool(match.group(1))))
    return rows
