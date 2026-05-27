"""Summarize markdown structure in unit content by source."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_LIST_RE = re.compile(r"^\s{0,3}(?:[-+*]\s+|\d+[.)]\s+)")
_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)|https?://\S+")


def summarize_unit_content_structure(units: Iterable[Any]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    total_units = 0
    for unit in units:
        total_units += 1
        grouped[_source(unit)].append(unit)

    rows = [_row(source, grouped[source]) for source in sorted(grouped, key=sort_key)]
    return {"total_units": total_units, "rows": rows, "source_summaries": rows}


def _row(source: str, units: list[Any]) -> dict[str, Any]:
    structures = [_structure(_content(unit)) for unit in units]
    heading_total = sum(item["heading_count"] for item in structures)
    return {
        "source": source,
        "unit_count": len(units),
        "heading_count": heading_total,
        "list_count": sum(item["list_count"] for item in structures),
        "table_count": sum(item["table_count"] for item in structures),
        "code_block_count": sum(item["code_block_count"] for item in structures),
        "link_count": sum(item["link_count"] for item in structures),
        "units_with_headings": sum(1 for item in structures if item["heading_count"]),
        "units_with_lists": sum(1 for item in structures if item["has_list"]),
        "units_with_tables": sum(1 for item in structures if item["has_table"]),
        "units_with_code_blocks": sum(1 for item in structures if item["has_code_block"]),
        "units_with_links": sum(1 for item in structures if item["link_count"]),
        "average_heading_count": f"{(heading_total / len(units)) if units else 0:.2f}",
    }


def _structure(content: str) -> dict[str, Any]:
    visible_lines: list[str] = []
    has_code_block = False
    in_fence = False
    for line in content.splitlines():
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            has_code_block = True
            in_fence = not in_fence
            continue
        if not in_fence:
            visible_lines.append(line)

    return {
        "heading_count": sum(1 for line in visible_lines if _HEADING_RE.match(line)),
        "list_count": sum(1 for line in visible_lines if _LIST_RE.match(line)),
        "table_count": sum(1 for line in visible_lines if _is_table_line(line)),
        "code_block_count": content.count("```") // 2 + content.count("~~~") // 2,
        "link_count": sum(1 for line in visible_lines for _match in _LINK_RE.finditer(line)),
        "has_list": any(_LIST_RE.match(line) for line in visible_lines),
        "has_table": any(_is_table_line(line) for line in visible_lines),
        "has_code_block": has_code_block,
    }


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _source(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "source_project") or meta.get("source") or meta.get("source_project")) or "unknown"


def _content(unit: Any) -> str:
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)
