"""Summarize Setext-style Markdown headings."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def summarize_unit_markdown_setext_headings(units: Iterable[Any], sample_limit: int = 10) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = count = 0
    levels: Counter[int] = Counter()
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        found = False
        previous: tuple[int, str] | None = None
        for line_number, line in _content_lines(str(get(unit, "content") or "")):
            level = _underline_level(line)
            if level and previous and previous[1].strip():
                found = True
                count += 1
                levels[level] += 1
                if len(samples) < limit:
                    samples.append({"unit_id": uid, "line_number": previous[0], "level": level, "text": previous[1].strip()})
                previous = None
                continue
            previous = (line_number, line) if line.strip() else None
        if found:
            units_with += 1
    return {"total_units": total, "units_with_setext_headings": units_with, "setext_heading_count": count, "level_counts": dict(sorted(levels.items())), "setext_heading_samples": samples}


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows


def _underline_level(line: str) -> int:
    text = line.strip()
    if len(text) < 2:
        return 0
    if set(text) == {"="}:
        return 1
    if set(text) == {"-"}:
        return 2
    return 0
