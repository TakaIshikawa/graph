"""CSV data quality rollup by source project."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, inline_text, metadata, render_csv, sort_key, write_csv

_FIELDNAMES = [
    "source_project",
    "unit_count",
    "missing_title_count",
    "missing_content_count",
    "missing_metadata_count",
    "missing_tags_count",
    "complete_unit_count",
    "quality_score",
]


def export_source_quality_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = _rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "source_count": len(rows), "bytes_written": bytes_written}


def _rows(units: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in units:
        groups[_source_project(unit)].append(unit)

    rows: list[dict[str, str | int]] = []
    for source, group in sorted(groups.items(), key=lambda item: sort_key(item[0])):
        missing_title = sum(1 for unit in group if not inline_text(get(unit, "title")))
        missing_content = sum(1 for unit in group if not inline_text(get(unit, "content")))
        missing_metadata = sum(1 for unit in group if not metadata(unit))
        missing_tags = sum(1 for unit in group if not _tags(unit))
        issue_count = missing_title + missing_content + missing_metadata + missing_tags
        possible_count = len(group) * 4
        complete_units = sum(1 for unit in group if _is_complete_unit(unit))
        rows.append(
            {
                "source_project": source,
                "unit_count": len(group),
                "missing_title_count": missing_title,
                "missing_content_count": missing_content,
                "missing_metadata_count": missing_metadata,
                "missing_tags_count": missing_tags,
                "complete_unit_count": complete_units,
                "quality_score": str(round((possible_count - issue_count) / possible_count, 3)) if possible_count else "0",
            }
        )
    return rows


def _source_project(unit: Mapping[str, Any] | object) -> str:
    return field_value(get(unit, "source_project")) or "Unknown"


def _tags(unit: Mapping[str, Any] | object) -> list[str]:
    raw = get(unit, "tags")
    values = raw if isinstance(raw, list | tuple | set) else []
    return [inline_text(tag) for tag in values if inline_text(tag)]


def _is_complete_unit(unit: Mapping[str, Any] | object) -> bool:
    return bool(inline_text(get(unit, "title")) and inline_text(get(unit, "content")) and metadata(unit) and _tags(unit))
