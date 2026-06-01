"""Summarize inline Markdown link title attributes in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\((\S+)(?:\s+((?:\"[^\"]*\"|'[^']*'|\([^)]*\))))\)")


def summarize_unit_markdown_link_title_attributes(units: Iterable[Any], high_density_threshold: int = 3, top_limit: int = 10) -> dict[str, Any]:
    unit_list = list(units)
    title_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    total = empty = 0
    for unit in unit_list:
        titles = _titles(str(get(unit, "content") or ""))
        if not titles:
            continue
        clean = [field_value(title) for title in titles]
        title_counts.update(title for title in clean if title)
        total += len(clean)
        empty_count = sum(1 for title in clean if not title)
        empty += empty_count
        if len(clean) >= high_density_threshold:
            rows.append({"unit_id": unit_id(unit), "title_attribute_count": len(clean), "empty_title_count": empty_count})
    repeated = [{"title_text": title, "count": count} for title, count in sorted(title_counts.items(), key=lambda item: (-item[1], sort_key(item[0]))) if count > 1]
    return {
        "total_units": len(unit_list),
        "units_with_title_attributes": sum(1 for unit in unit_list if _titles(str(get(unit, "content") or ""))),
        "title_attribute_count": total,
        "empty_title_attribute_count": empty,
        "repeated_title_text": repeated[: max(0, top_limit)],
        "high_density_units": sorted(rows, key=lambda row: (-row["title_attribute_count"], sort_key(row["unit_id"]))),
    }


def _titles(content: str) -> list[str]:
    rows: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _LINK_RE.finditer(line):
            rows.append(_strip_title(match.group(3)))
    return rows


def _strip_title(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and ((text[0] == text[-1] and text[0] in {"'", '"'}) or (text[0] == "(" and text[-1] == ")")):
        return text[1:-1]
    return text
