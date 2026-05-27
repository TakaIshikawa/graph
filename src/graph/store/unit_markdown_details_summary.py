"""Summarize HTML details blocks in Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DETAILS_OPEN_RE = re.compile(r"<details\b([^>]*)>", re.IGNORECASE)
_DETAILS_CLOSE_RE = re.compile(r"</details\s*>", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary\b[^>]*>(.*?)</summary\s*>", re.IGNORECASE)
_OPEN_ATTR_RE = re.compile(r"(?:^|\s)open(?:\s|=|$)", re.IGNORECASE)


def summarize_unit_markdown_details(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = details = open_count = missing_summary = unclosed = 0
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        blocks = _blocks(str(get(unit, "content") or ""))
        if blocks:
            units_with += 1
        for block in blocks:
            details += 1
            open_count += 1 if block["is_open"] else 0
            missing_summary += 1 if not block["summary"] else 0
            unclosed += 1 if block["unclosed"] else 0
            if len(samples) < limit:
                samples.append({"unit_id": unit_id(unit), "start_line": block["start_line"], "summary": block["summary"], "is_open": block["is_open"]})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["start_line"])))
    return {
        "total_units": total,
        "units_with_details": units_with,
        "details_count": details,
        "open_details_count": open_count,
        "missing_summary_count": missing_summary,
        "unclosed_details_count": unclosed,
        "samples": samples[:limit],
    }


def _blocks(content: str) -> list[dict[str, Any]]:
    lines = _content_lines(content)
    rows: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    for line_number, line in lines:
        if active is None:
            match = _DETAILS_OPEN_RE.search(line)
            if match:
                active = {"start_line": line_number, "summary": "", "is_open": bool(_OPEN_ATTR_RE.search(match.group(1)))}
        if active is None:
            continue
        summary = _SUMMARY_RE.search(line)
        if summary and not active["summary"]:
            active["summary"] = field_value(summary.group(1))
        if _DETAILS_CLOSE_RE.search(line):
            rows.append({**active, "unclosed": False})
            active = None
    if active is not None:
        rows.append({**active, "unclosed": True})
    return rows


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
