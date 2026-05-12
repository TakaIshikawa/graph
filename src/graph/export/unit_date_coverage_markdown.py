"""Markdown export for unit date field coverage."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_DATE_FIELDS = ["created_at", "ingested_at", "updated_at"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_date_coverage_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report for unit date coverage."""
    unit_list = sorted(list(units), key=_unit_sort_key)
    markdown = _render(unit_list)

    if path is None:
        return markdown

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_project_count": len({_unit_source(unit) for unit in unit_list}),
        "bytes_written": output_path.stat().st_size,
    }


def _render(units: list[KnowledgeUnit]) -> str:
    lines = [
        "# Unit Date Coverage",
        "",
        "## Overall",
        "",
        *_table(_field_stats(units)),
    ]

    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[_unit_source(unit)].append(unit)

    for source_project in sorted(groups, key=_sort_key):
        lines.extend(
            [
                "",
                f"## {_markdown_cell(source_project)}",
                "",
                *_table(_field_stats(groups[source_project])),
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _field_stats(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    total = len(units)
    for field in _DATE_FIELDS:
        values = [_date_value(getattr(unit, field, None)) for unit in units]
        present = sorted(value for value in values if value)
        present_count = len(present)
        rows.append(
            {
                "field": field,
                "present_count": present_count,
                "missing_count": total - present_count,
                "coverage_percent": _decimal(present_count * 100 / total) if total else "0.00",
                "earliest": present[0] if present else "",
                "latest": present[-1] if present else "",
            }
        )
    return rows


def _table(rows: list[dict[str, str | int]]) -> list[str]:
    lines = [
        "| Field | Present | Missing | Coverage | Earliest | Latest |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{_markdown_cell(row['field'])} | "
            f"{row['present_count']} | "
            f"{row['missing_count']} | "
            f"{row['coverage_percent']} | "
            f"{_markdown_cell(row['earliest'])} | "
            f"{_markdown_cell(row['latest'])} |"
        )
    return lines


def _date_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime | date):
        return value.isoformat()
    return _inline_text(value)


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_source(unit)), _sort_key(unit.title), _sort_key(unit.id or unit.source_id))


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")


def _decimal(value: float) -> str:
    return f"{value:.2f}"
