"""Markdown activity report grouped by source project and entity type."""

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
class _SourceEntityActivity:
    source_project: str
    source_entity_type: str
    units: tuple[KnowledgeUnit, ...]

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def first_created_at(self) -> datetime | None:
        values = [unit.created_at for unit in self.units if isinstance(unit.created_at, datetime)]
        return min(values) if values else None

    @property
    def newest_updated_at(self) -> datetime | None:
        values = [unit.updated_at for unit in self.units if isinstance(unit.updated_at, datetime)]
        return max(values) if values else None

    @property
    def tag_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for unit in self.units:
            for tag in _unit_tags(unit):
                counts[tag] += 1
        return counts

    @property
    def tag_count(self) -> int:
        return len(self.tag_counts)


def export_source_entity_activity_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_count: int = 1,
    sample_limit: int = 3,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown source/entity activity report."""
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    if (
        not isinstance(sample_limit, int)
        or isinstance(sample_limit, bool)
        or sample_limit < 0
    ):
        raise ValueError("sample_limit must be a non-negative integer")

    unit_list = sorted(list(units), key=_unit_sort_key)
    activities = _activities(unit_list, min_count=min_count)
    text = _render_report(
        activities,
        units_scanned=len(unit_list),
        min_count=min_count,
        sample_limit=sample_limit,
    )

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(unit_list),
        "groups_exported": len(activities),
        "min_count": min_count,
        "sample_limit": sample_limit,
        "bytes_written": output_path.stat().st_size,
    }


def _activities(units: list[KnowledgeUnit], *, min_count: int) -> list[_SourceEntityActivity]:
    grouped: dict[tuple[str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        grouped[(_unit_source(unit), _entity_type(unit))].append(unit)

    activities = [
        _SourceEntityActivity(source, entity_type, tuple(group_units))
        for (source, entity_type), group_units in grouped.items()
        if len(group_units) >= min_count
    ]
    return sorted(
        activities,
        key=lambda activity: (-activity.unit_count, activity.source_project.casefold(), activity.source_entity_type.casefold()),
    )


def _render_report(
    activities: list[_SourceEntityActivity],
    *,
    units_scanned: int,
    min_count: int,
    sample_limit: int,
) -> str:
    lines = [
        "# Source Entity Activity",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Groups reported | {len(activities)} |",
        f"| Min count | {min_count} |",
        f"| Sample limit | {sample_limit} |",
        "",
        "## Groups",
        "",
        "| Source project | Entity type | Units | First created | Newest updated | Tags | Top tags | Samples |",
        "| --- | --- | ---: | --- | --- | ---: | --- | --- |",
    ]
    if activities:
        for activity in activities:
            lines.append(
                "| "
                f"{_markdown_cell(activity.source_project)} | "
                f"{_markdown_cell(activity.source_entity_type)} | "
                f"{activity.unit_count} | "
                f"{_markdown_cell(_datetime_text(activity.first_created_at))} | "
                f"{_markdown_cell(_datetime_text(activity.newest_updated_at))} | "
                f"{activity.tag_count} | "
                f"{_markdown_cell(_top_tags_text(activity.tag_counts))} | "
                f"{_markdown_cell(_samples_text(activity.units, limit=sample_limit))} |"
            )
    else:
        lines.append("| _None_ | _None_ | 0 | _None_ | _None_ | 0 | _None_ | _None_ |")
    return "\n".join(lines).rstrip() + "\n"


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)


def _top_tags_text(counter: Counter[str], *, limit: int = 5) -> str:
    if not counter:
        return "_None_"
    tags = sorted(counter.items(), key=lambda item: (-item[1], _sort_key(item[0])))[:limit]
    return "; ".join(f"{tag} ({count})" for tag, count in tags)


def _samples_text(units: tuple[KnowledgeUnit, ...], *, limit: int) -> str:
    if limit == 0:
        return "_None_"
    labels = [_unit_label(unit) for unit in sorted(units, key=_unit_sort_key)[:limit]]
    return "; ".join(labels) if labels else "_None_"


def _unit_source(unit: KnowledgeUnit) -> str:
    return _inline_text(getattr(unit.source_project, "value", unit.source_project)) or "Unknown"


def _entity_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or "Unknown"


def _unit_label(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.title) or _inline_text(unit.source_id) or _inline_text(unit.id) or "Untitled"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str, str]:
    return (
        _unit_source(unit),
        _entity_type(unit),
        _inline_text(unit.created_at.isoformat() if isinstance(unit.created_at, datetime) else unit.created_at),
        _inline_text(unit.source_id),
        _inline_text(unit.id),
    )


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
