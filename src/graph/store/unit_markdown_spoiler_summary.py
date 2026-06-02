"""Summarize Markdown spoiler markers in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_PIPE_RE = re.compile(r"(?<!\|)\|\|(.+?)\|\|(?!\|)")
_DETAILS_RE = re.compile(r"<details\b", re.IGNORECASE)


def summarize_unit_markdown_spoilers(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = spoiler_count = details_count = pipe_count = 0
    rows: list[dict[str, int | str]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_pipe = unit_details = 0
        for line in _content_lines(str(get(unit, "content") or "")):
            unit_pipe += len(_PIPE_RE.findall(line))
            unit_details += len(_DETAILS_RE.findall(line))
        unit_total = unit_pipe + unit_details
        if unit_total:
            units_with += 1
            spoiler_count += unit_total
            details_count += unit_details
            pipe_count += unit_pipe
            rows.append({"unit_id": uid, "pipe_spoiler_count": unit_pipe, "details_count": unit_details, "spoiler_count": unit_total})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {
        "total_units": total,
        "units_with_spoilers": units_with,
        "spoiler_count": spoiler_count,
        "details_count": details_count,
        "pipe_spoiler_count": pipe_count,
        "units": rows,
    }


def _content_lines(content: str) -> list[str]:
    rows: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append(line)
    return rows
