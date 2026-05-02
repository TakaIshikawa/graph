"""Markdown source coverage reports for knowledge units."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _SourceGroup:
    source_project: str
    source_entity_type: str
    units: tuple[KnowledgeUnit, ...]

    @property
    def oldest_updated_at(self) -> datetime | None:
        timestamps = [
            unit.updated_at for unit in self.units if isinstance(unit.updated_at, datetime)
        ]
        return min(timestamps) if timestamps else None

    @property
    def newest_updated_at(self) -> datetime | None:
        timestamps = [
            unit.updated_at for unit in self.units if isinstance(unit.updated_at, datetime)
        ]
        return max(timestamps) if timestamps else None

    @property
    def tag_counts(self) -> Counter[str]:
        counter: Counter[str] = Counter()
        for unit in self.units:
            counter.update(_unit_tags(unit))
        return counter


def export_source_coverage_markdown(
    units: Iterable[KnowledgeUnit],
    *,
    stale_after_days: int | None = None,
    as_of: datetime | None = None,
    top_tags_limit: int = 5,
) -> str:
    """Return a deterministic Markdown report of unit coverage by source."""
    if stale_after_days is not None and (
        not isinstance(stale_after_days, int)
        or isinstance(stale_after_days, bool)
        or stale_after_days < 0
    ):
        raise ValueError("stale_after_days must be a non-negative integer or None")
    if (
        not isinstance(top_tags_limit, int)
        or isinstance(top_tags_limit, bool)
        or top_tags_limit < 0
    ):
        raise ValueError("top_tags_limit must be a non-negative integer")

    all_units = sorted(list(units), key=_unit_sort_key)
    groups = _source_groups(all_units)
    newest_update = _newest_updated_at(all_units)
    reference_time = as_of or newest_update
    stale_cutoff = (
        reference_time - timedelta(days=stale_after_days)
        if stale_after_days is not None and reference_time is not None
        else None
    )

    return _render_report(
        groups,
        units_scanned=len(all_units),
        oldest_updated_at=_oldest_updated_at(all_units),
        newest_updated_at=newest_update,
        stale_requested=stale_after_days is not None,
        stale_cutoff=stale_cutoff,
        top_tags_limit=top_tags_limit,
    )


def _source_groups(units: list[KnowledgeUnit]) -> list[_SourceGroup]:
    grouped: dict[tuple[str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        grouped[(_field_value(unit.source_project), _inline_text(unit.source_entity_type))].append(
            unit
        )
    return [
        _SourceGroup(source_project, source_entity_type, tuple(group_units))
        for (source_project, source_entity_type), group_units in sorted(
            grouped.items(), key=lambda item: _group_sort_key(*item[0])
        )
    ]


def _render_report(
    groups: list[_SourceGroup],
    *,
    units_scanned: int,
    oldest_updated_at: datetime | None,
    newest_updated_at: datetime | None,
    stale_requested: bool,
    stale_cutoff: datetime | None,
    top_tags_limit: int,
) -> str:
    lines = [
        "# Source Coverage Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Source groups | {len(groups)} |",
        f"| Oldest updated | {_datetime_text(oldest_updated_at)} |",
        f"| Newest updated | {_datetime_text(newest_updated_at)} |",
        "",
        "## Source Groups",
        "",
    ]

    header = "| Source project | Entity type | Units | Oldest updated | Newest updated | Top tags |"
    separator = "| --- | --- | ---: | --- | --- | --- |"
    if stale_requested:
        header = (
            "| Source project | Entity type | Units | Oldest updated | Newest updated | "
            "Top tags | Status |"
        )
        separator = "| --- | --- | ---: | --- | --- | --- | --- |"
    lines.extend([header, separator])

    if groups:
        for group in groups:
            row = (
                "| "
                f"{_markdown_cell(group.source_project)} | "
                f"{_markdown_cell(group.source_entity_type or '_None_')} | "
                f"{len(group.units)} | "
                f"{_datetime_text(group.oldest_updated_at)} | "
                f"{_datetime_text(group.newest_updated_at)} | "
                f"{_markdown_cell(_top_tags_text(group.tag_counts, top_tags_limit))} |"
            )
            if stale_requested:
                status = (
                    _stale_status(group, stale_cutoff) if stale_cutoff is not None else "_None_"
                )
                row = f"{row[:-1]}| {status} |"
            lines.append(row)
    else:
        empty_row = "| _None_ | _None_ | 0 | _None_ | _None_ | _None_ |"
        if stale_requested:
            empty_row = f"{empty_row[:-1]}| _None_ |"
        lines.append(empty_row)

    if stale_requested:
        lines.extend(
            [
                "",
                "## Stale Sources",
                "",
            ]
        )
        stale_groups = (
            [group for group in groups if _is_stale(group, stale_cutoff)]
            if stale_cutoff is not None
            else []
        )
        if stale_groups:
            for group in stale_groups:
                lines.append(
                    "- "
                    f"{_inline_markdown(group.source_project)} / "
                    f"{_inline_markdown(group.source_entity_type or '_None_')}: "
                    f"{len(group.units)} units, newest update "
                    f"{_datetime_text(group.newest_updated_at)}"
                )
        else:
            lines.append("_No stale source groups._")

    return "\n".join(lines).rstrip() + "\n"


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted(
        {_inline_text(tag) for tag in unit.tags if _inline_text(tag)},
        key=lambda tag: (tag.casefold(), tag),
    )


def _top_tags_text(counter: Counter[str], limit: int) -> str:
    if not counter or limit == 0:
        return "_None_"
    return ", ".join(
        f"{tag} ({count})"
        for tag, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    )


def _stale_status(group: _SourceGroup, stale_cutoff: datetime) -> str:
    return "Stale" if _is_stale(group, stale_cutoff) else "Current"


def _is_stale(group: _SourceGroup, stale_cutoff: datetime) -> bool:
    newest = group.newest_updated_at
    return newest is None or newest < stale_cutoff


def _oldest_updated_at(units: list[KnowledgeUnit]) -> datetime | None:
    timestamps = [unit.updated_at for unit in units if isinstance(unit.updated_at, datetime)]
    return min(timestamps) if timestamps else None


def _newest_updated_at(units: list[KnowledgeUnit]) -> datetime | None:
    timestamps = [unit.updated_at for unit in units if isinstance(unit.updated_at, datetime)]
    return max(timestamps) if timestamps else None


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str, str]:
    return (
        _field_value(unit.source_project),
        _inline_text(unit.source_entity_type),
        _datetime_text(unit.updated_at),
        _inline_text(unit.source_id),
        _inline_text(unit.id),
    )


def _group_sort_key(source_project: str, source_entity_type: str) -> tuple[str, str, str, str]:
    return (
        source_project.casefold(),
        source_project,
        source_entity_type.casefold(),
        source_entity_type,
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


def _inline_markdown(value: object) -> str:
    return (
        _inline_text(value)
        .replace("\\", r"\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")
