"""Summarize body Markdown hashtags."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
_TAG_RE = re.compile(r"(?<![\w/&?=])#([A-Za-z][A-Za-z0-9_-]*(?:/[A-Za-z0-9_-]+)*)")
_CODE_RE = re.compile(r"`+[^`\n]*`+")


def summarize_unit_markdown_tags(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = max_depth = 0
    tag_counts: Counter[str] = Counter()
    nested_counts: Counter[str] = Counter()
    for unit in units:
        tags = _tags(str(get(unit, "content") or ""))
        if tags:
            units_with += 1
        for tag in tags:
            normalized = f"#{tag}".casefold()
            depth = normalized.count("/") + 1
            total += 1; max_depth = max(max_depth, depth); tag_counts[normalized] += 1
            if depth > 1:
                nested_counts[normalized] += 1
    return {"total_tags": total, "units_with_tags": units_with, "tag_counts": dict(sorted(tag_counts.items())), "nested_tag_counts": dict(sorted(nested_counts.items())), "max_tag_depth": max_depth}


def _tags(content: str) -> list[str]:
    rows: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or _HEADING_RE.match(line):
            continue
        rows.extend(match.group(1) for match in _TAG_RE.finditer(_CODE_RE.sub("", line)))
    return rows
