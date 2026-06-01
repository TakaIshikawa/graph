"""Summarize backslash-escaped Markdown punctuation in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ESCAPE_RE = re.compile(r"\\([\\`*{}\[\]()#+\-.!_>~|=])")


def summarize_unit_markdown_escaped_characters(units: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Count escaped Markdown punctuation characters outside fenced code."""
    unit_list = list(units)
    escaped_character_counts: Counter[str] = Counter()
    total = 0
    affected: set[str] = set()
    examples: list[dict[str, str | int]] = []
    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        for line_number, character in _escaped_characters(_content(unit)):
            total += 1
            escaped_character_counts[character] += 1
            affected.add(uid)
            examples.append({"unit_id": uid, "line_number": line_number, "escaped_character": character})
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["escaped_character"])))
    return {
        "total_units": len(unit_list),
        "escaped_character_count": total,
        "escaped_character_counts": dict(sorted(escaped_character_counts.items(), key=lambda item: sort_key(item[0]))),
        "affected_units": sorted(affected, key=sort_key),
        "examples": examples[:sample_limit],
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _escaped_characters(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        rows.extend((line_number, match.group(1)) for match in _ESCAPE_RE.finditer(line))
    return rows
