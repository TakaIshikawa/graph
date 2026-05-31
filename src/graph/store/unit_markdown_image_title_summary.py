"""Summarize Markdown image title attributes in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_IMAGE_RE = re.compile(r"!\[([^\]\n]*)]\(([^)\n]*)\)")
_TITLE_RE = re.compile(r"""^\s*\S+(?:\s+(?:"([^"]*)"|'([^']*)'|\(([^)]*)\)))?\s*$""")


def summarize_unit_markdown_image_titles(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = with_title = without_title = 0
    title_counts: Counter[str] = Counter()
    title_originals: dict[str, str] = {}
    examples: list[dict[str, str | int]] = []
    units_with = set()
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, alt, title in _images(str(get(unit, "content") or "")):
            units_with.add(uid)
            if title:
                with_title += 1
                key = title.casefold()
                title_counts[key] += 1
                title_originals.setdefault(key, title)
            else:
                without_title += 1
            if len(examples) < limit:
                examples.append({"unit_id": uid, "line_number": line_number, "alt_text": alt, "title": title})
    duplicate_titles = [
        {"title": title_originals[key], "image_count": count}
        for key, count in title_counts.items()
        if count > 1
    ]
    duplicate_titles.sort(key=lambda row: (-int(row["image_count"]), sort_key(row["title"])))
    return {"total_units": total, "units_with_images": len(units_with), "with_title": with_title, "without_title": without_title, "duplicate_titles": duplicate_titles, "examples": examples[:limit]}


def _images(content: str) -> list[tuple[int, str, str]]:
    rows = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _IMAGE_RE.finditer(line):
            title_match = _TITLE_RE.match(match.group(2))
            title = ""
            if title_match:
                title = field_value(next((group for group in title_match.groups() if group is not None), ""))
            rows.append((line_number, field_value(match.group(1)), title))
    return rows
