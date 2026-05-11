"""Markdown timeline export helpers for dated knowledge units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, overload

from graph.export.timelinejs import (
    _clean_text,
    _date_sort_value,
    _first_text,
    _scalar_text,
    _unit_sort_key,
    _unit_start_date,
)
from graph.types.models import KnowledgeUnit

SOURCE_URL_METADATA_KEYS = ("source_url", "external_url")


@overload
def export_units_to_markdown_timeline(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: None = None,
) -> str: ...


@overload
def export_units_to_markdown_timeline(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path,
) -> dict[str, Any]: ...


def export_units_to_markdown_timeline(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a chronological Markdown timeline for dated units."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    entries: list[tuple[str, tuple[str, str, str], KnowledgeUnit]] = []
    skipped_count = 0

    for unit in unit_list:
        start = _unit_start_date(unit)
        if start is None:
            skipped_count += 1
            continue
        entries.append((_date_sort_value(start), _unit_sort_key(unit), unit))

    entries.sort(key=lambda entry: (entry[0], entry[1]))
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for sort_date, _, unit in entries:
        grouped[_heading_date(sort_date)].append(unit)

    lines = ["# Timeline", ""]
    for heading in sorted(grouped):
        lines.extend([f"## {heading}", ""])
        for unit in grouped[heading]:
            lines.extend(_unit_lines(unit))
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(entries),
        "skipped_count": skipped_count,
        "bytes_written": output_path.stat().st_size,
    }


def _heading_date(sort_date: str) -> str:
    return sort_date[:10]


def _unit_lines(unit: KnowledgeUnit) -> list[str]:
    title = _clean_text(unit.title) or "Untitled graph unit"
    lines = [f"- **{title}**"]
    details = [_clean_text(_scalar_text(unit.source_project))]
    tags = sorted(tag for tag in (_clean_text(_scalar_text(tag)) for tag in unit.tags) if tag)
    if tags:
        details.append("tags: " + ", ".join(tags))
    url = _first_text(unit.metadata, SOURCE_URL_METADATA_KEYS)
    if url:
        details.append(f"[source]({url})")
    lines.append(f"  - {' | '.join(detail for detail in details if detail)}")
    return lines
