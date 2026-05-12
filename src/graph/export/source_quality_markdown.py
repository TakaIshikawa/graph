"""Markdown data quality report by source project."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _SourceQuality:
    source: str
    units: tuple[KnowledgeUnit, ...]

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def missing_title_count(self) -> int:
        return sum(1 for unit in self.units if not _inline_text(unit.title))

    @property
    def missing_content_count(self) -> int:
        return sum(1 for unit in self.units if not _inline_text(unit.content))

    @property
    def missing_metadata_count(self) -> int:
        return sum(1 for unit in self.units if not (unit.metadata or {}))

    @property
    def missing_tags_count(self) -> int:
        return sum(1 for unit in self.units if not [tag for tag in unit.tags if _inline_text(tag)])

    @property
    def average_content_length(self) -> int:
        if not self.units:
            return 0
        return round(sum(len(_inline_text(unit.content)) for unit in self.units) / len(self.units))

    @property
    def newest_updated_at(self) -> datetime | None:
        timestamps = [unit.updated_at for unit in self.units if isinstance(unit.updated_at, datetime)]
        return max(timestamps) if timestamps else None


def export_source_quality_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a Markdown quality report grouped by source project."""
    unit_list = sorted(list(units), key=_unit_sort_key)
    groups = _source_groups(unit_list)
    text = _render_report(groups, units_scanned=len(unit_list))

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_count": len(groups),
        "bytes_written": output_path.stat().st_size,
    }


def _source_groups(units: list[KnowledgeUnit]) -> list[_SourceQuality]:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        grouped[_unit_source(unit)].append(unit)
    return [
        _SourceQuality(source, tuple(group_units))
        for source, group_units in sorted(grouped.items(), key=lambda item: _sort_key(item[0]))
    ]


def _render_report(groups: list[_SourceQuality], *, units_scanned: int) -> str:
    totals = _SourceQuality("Total", tuple(unit for group in groups for unit in group.units))
    lines = [
        "# Source Quality Report",
        "",
        "## Totals",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Source projects | {len(groups)} |",
        f"| Missing titles | {totals.missing_title_count} |",
        f"| Missing content | {totals.missing_content_count} |",
        f"| Missing metadata | {totals.missing_metadata_count} |",
        f"| Missing tags | {totals.missing_tags_count} |",
        f"| Average content length | {totals.average_content_length} |",
        f"| Newest updated | {_datetime_text(totals.newest_updated_at)} |",
        "",
        "## Sources",
        "",
        "| Source project | Units | Missing title | Missing content | Missing metadata | Missing tags | Avg content length | Newest updated |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if groups:
        for group in groups:
            lines.append(
                "| "
                f"{_markdown_cell(group.source)} | "
                f"{group.unit_count} | "
                f"{group.missing_title_count} | "
                f"{group.missing_content_count} | "
                f"{group.missing_metadata_count} | "
                f"{group.missing_tags_count} | "
                f"{group.average_content_length} | "
                f"{_markdown_cell(_datetime_text(group.newest_updated_at))} |"
            )
    else:
        lines.append("| _None_ | 0 | 0 | 0 | 0 | 0 | 0 | _None_ |")
    return "\n".join(lines).rstrip() + "\n"


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (
        _unit_source(unit),
        _inline_text(unit.title),
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
