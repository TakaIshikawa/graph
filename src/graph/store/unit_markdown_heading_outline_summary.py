"""Summarize Markdown heading outlines."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ATX_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_SPACE_RE = re.compile(r"\s+")


def summarize_unit_markdown_heading_outlines(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = units_with = total_headings = max_depth = skipped = duplicate_text = 0
    samples: list[dict[str, Any]] = []
    for unit in units:
        total_units += 1
        headings = _headings(str(get(unit, "content") or ""))
        if not headings:
            continue
        units_with += 1
        total_headings += len(headings)
        max_depth = max(max_depth, max(level for level, _text, _line in headings))
        skipped += _skipped_levels(headings)
        duplicate_text += sum(1 for count in Counter(_normalize(text) for _level, text, _line in headings).values() if count > 1)
        if len(samples) < sample_limit:
            samples.append({"unit_id": unit_id(unit), "headings": [{"level": level, "text": text, "line": line} for level, text, line in headings[:sample_limit]]})
    return {"total_units": total_units, "units_with_headings": units_with, "total_headings": total_headings, "max_heading_depth": max_depth, "skipped_level_issue_count": skipped, "duplicate_heading_text_count": duplicate_text, "samples": samples}


def _headings(content: str) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    in_fence = False
    previous = ""
    previous_line = 0
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            previous = ""
            continue
        if in_fence:
            continue
        if match := _ATX_RE.match(line):
            rows.append((len(match.group(1)), match.group(2).strip(), line_number))
        elif previous.strip() and (setext := _SETEXT_RE.match(line)):
            rows.append((1 if setext.group(1).startswith("=") else 2, previous.strip(), previous_line))
        previous = line
        previous_line = line_number
    return rows


def _skipped_levels(headings: list[tuple[int, str, int]]) -> int:
    issues = 0
    previous = 0
    for level, _text, _line in headings:
        if previous and level > previous + 1:
            issues += 1
        previous = level
    return issues


def _normalize(text: str) -> str:
    return _SPACE_RE.sub(" ", text.strip().casefold())
