"""Summarize block-level raw HTML tags in Markdown unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_BLOCK_TAGS = {"article", "aside", "details", "div", "figure", "footer", "form", "header", "iframe", "main", "nav", "section", "table"}
_TAG_RE = re.compile(r"^\s{0,3}</?([A-Za-z][A-Za-z0-9:-]*)(?:\s|>|/>)")


def summarize_unit_markdown_html_blocks(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = 0
    grouped: dict[str, dict[str, Any]] = {}
    for unit in units:
        total += 1
        uid = unit_id(unit)
        seen: set[str] = set()
        for line_number, tag, snippet in _blocks(str(get(unit, "content") or "")):
            row = grouped.setdefault(tag, {"tag": tag, "block_count": 0, "unit_ids": set(), "examples": []})
            row["block_count"] += 1
            seen.add(tag)
            if len(row["examples"]) < limit:
                row["examples"].append({"unit_id": uid, "line_number": line_number, "snippet": snippet})
        for tag in seen:
            grouped[tag]["unit_ids"].add(uid)
    tags = [
        {"tag": row["tag"], "block_count": row["block_count"], "unit_count": len(row["unit_ids"]), "examples": row["examples"][:limit]}
        for row in grouped.values()
    ]
    tags.sort(key=lambda row: (-int(row["block_count"]), sort_key(row["tag"])))
    return {"total_units": total, "html_blocks": tags}


def _blocks(content: str) -> list[tuple[int, str, str]]:
    rows = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _TAG_RE.match(line)
        if match and match.group(1).casefold() in _BLOCK_TAGS:
            rows.append((line_number, match.group(1).casefold(), field_value(line.strip())))
    return rows
