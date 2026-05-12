"""Markdown activity report grouped by tag."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _TagActivity:
    tag: str
    units: tuple[KnowledgeUnit, ...]

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def first_created_at(self) -> datetime | None:
        timestamps = [unit.created_at for unit in self.units if isinstance(unit.created_at, datetime)]
        return min(timestamps) if timestamps else None

    @property
    def newest_updated_at(self) -> datetime | None:
        timestamps = [unit.updated_at for unit in self.units if isinstance(unit.updated_at, datetime)]
        return max(timestamps) if timestamps else None

    @property
    def source_counts(self) -> Counter[str]:
        return Counter(_unit_source(unit) for unit in self.units)

    @property
    def source_count(self) -> int:
        return len(self.source_counts)


def export_tag_activity_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_count: int = 1,
    top_source_limit: int = 3,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report of tag activity windows."""
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    if not isinstance(top_source_limit, int) or isinstance(top_source_limit, bool) or top_source_limit < 1:
        raise ValueError("top_source_limit must be a positive integer")

    unit_list = sorted(list(units), key=_unit_sort_key)
    activities = _tag_activities(unit_list, min_count=min_count)
    text = _render_report(
        activities,
        units_scanned=len(unit_list),
        min_count=min_count,
        top_source_limit=top_source_limit,
    )

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(unit_list),
        "tag_count": len(activities),
        "bytes_written": output_path.stat().st_size,
    }


def _tag_activities(units: list[KnowledgeUnit], *, min_count: int) -> list[_TagActivity]:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        for tag in _unit_tags(unit):
            grouped[tag].append(unit)

    activities = [
        _TagActivity(tag, tuple(tag_units))
        for tag, tag_units in grouped.items()
        if len(tag_units) >= min_count
    ]
    return sorted(activities, key=lambda activity: (-activity.unit_count, _sort_key(activity.tag)))


def _render_report(
    activities: list[_TagActivity],
    *,
    units_scanned: int,
    min_count: int,
    top_source_limit: int,
) -> str:
    lines = [
        "# Tag Activity Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Tags reported | {len(activities)} |",
        f"| Min count | {min_count} |",
        "",
        "## Tags",
        "",
        "| Tag | Units | First created | Newest updated | Sources | Top sources |",
        "| --- | ---: | --- | --- | ---: | --- |",
    ]
    if activities:
        for activity in activities:
            lines.append(
                "| "
                f"{_markdown_cell(activity.tag)} | "
                f"{activity.unit_count} | "
                f"{_markdown_cell(_datetime_text(activity.first_created_at))} | "
                f"{_markdown_cell(_datetime_text(activity.newest_updated_at))} | "
                f"{activity.source_count} | "
                f"{_markdown_cell(_top_sources_text(activity.source_counts, limit=top_source_limit))} |"
            )
    else:
        lines.append("| _None_ | 0 | _None_ | _None_ | 0 | _None_ |")
    return "\n".join(lines).rstrip() + "\n"


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _top_sources_text(source_counts: Counter[str], *, limit: int) -> str:
    if not source_counts:
        return "_None_"
    sources = sorted(source_counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))[:limit]
    return "; ".join(f"{source} ({count})" for source, count in sources)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        _unit_source(unit),
        _inline_text(unit.source_id),
        _inline_text(unit.id),
    )


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _datetime_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    text = _inline_text(value)
    return text or "_None_"


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")
