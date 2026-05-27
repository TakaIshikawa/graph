"""CSV export for Markdown section lengths in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "heading_count", "section_count", "max_section_line_count", "average_section_line_count", "longest_section_heading"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


def export_units_to_section_length_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, int | str]:
    sections, headings = _sections("" if get(unit, "content") is None else str(get(unit, "content")))
    counts = [count for _heading, count in sections]
    longest_heading = max(sections, key=lambda item: (item[1], sort_key(item[0])))[0] if sections else ""
    return {
        "unit_id": unit_id(unit),
        "heading_count": len(headings),
        "section_count": len(sections),
        "max_section_line_count": max(counts, default=0),
        "average_section_line_count": f"{(sum(counts) / len(counts)):.2f}" if counts else "0.00",
        "longest_section_heading": longest_heading,
    }


def _sections(content: str) -> tuple[list[tuple[str, int]], list[str]]:
    sections: list[tuple[str, int]] = []
    headings: list[str] = []
    current_heading = ""
    current_lines = 0
    preheading_has_content = False
    fence = ""
    for line in content.splitlines():
        match = _FENCE_RE.match(line)
        marker = match.group(1) if match else ""
        if marker and not fence:
            fence = marker[0]
            current_lines += 1
            preheading_has_content = preheading_has_content or bool(line.strip())
            continue
        if fence:
            current_lines += 1
            if line.lstrip().startswith(fence * 3):
                fence = ""
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            title = field_value(heading.group(2))
            if headings or preheading_has_content:
                sections.append((current_heading, current_lines))
            headings.append(title)
            current_heading = title
            current_lines = 0
            preheading_has_content = False
            continue
        current_lines += 1
        preheading_has_content = preheading_has_content or bool(line.strip())
    if headings or preheading_has_content:
        sections.append((current_heading, current_lines))
    return sections, headings
