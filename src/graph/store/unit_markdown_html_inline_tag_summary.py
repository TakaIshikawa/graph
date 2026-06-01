"""Summarize inline HTML tags embedded in Markdown text."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9:-]*)(?:\s[^<>]*)?/?>")
_BLOCK_LINE_RE = re.compile(r"^\s{0,3}</?([A-Za-z][A-Za-z0-9:-]*)(?:\s|>|/>)\s*$")


def summarize_unit_markdown_html_inline_tags(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize raw HTML tags that appear inline in Markdown paragraphs."""
    limit = max(0, sample_limit)
    total = count = 0
    tags: Counter[str] = Counter()
    affected: set[str] = set()
    examples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, tag in _inline_tags(str(get(unit, "content") or "")):
            count += 1
            tags[tag] += 1
            affected.add(uid)
            examples.append({"unit_id": uid, "line": line_number, "tag": tag})
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["tag"])))
    return {
        "total_units": total,
        "inline_tag_count": count,
        "affected_units": len(affected),
        "tag_counts": {tag: tags[tag] for tag in sorted(tags, key=sort_key)},
        "examples": examples[:limit],
    }


def _inline_tags(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or "<!--" in line or _BLOCK_LINE_RE.match(line):
            continue
        for match in _TAG_RE.finditer(line):
            rows.append((line_number, match.group(1).casefold()))
    return rows
