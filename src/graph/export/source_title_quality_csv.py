"""CSV export for source title quality summaries."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "total_units",
    "blank_title_count",
    "blank_title_rate",
    "duplicate_title_count",
    "duplicate_title_rate",
    "title_content_duplicate_count",
    "title_content_duplicate_rate",
    "very_short_title_count",
    "very_short_title_rate",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_title_quality_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write title quality counts by source/type."""
    unit_list = list(units)
    rows = _quality_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _quality_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[(_unit_source(unit), _unit_source_type(unit))].append(unit)

    rows: list[dict[str, str | int]] = []
    for source_project, source_entity_type in sorted(
        groups,
        key=lambda key: (_sort_key(key[0]), _sort_key(key[1])),
    ):
        group_units = groups[(source_project, source_entity_type)]
        total_units = len(group_units)
        normalized_titles = [_normalized_title(unit.title) for unit in group_units]
        title_counts = Counter(title for title in normalized_titles if title)
        duplicate_title_count = sum(1 for title in normalized_titles if title and title_counts[title] > 1)
        blank_title_count = sum(1 for title in normalized_titles if not title)
        title_content_duplicate_count = sum(
            1
            for unit, title in zip(group_units, normalized_titles, strict=False)
            if title and title == _normalized_title(unit.content)
        )
        very_short_title_count = sum(
            1 for unit in group_units if 0 < len(_inline_text(unit.title)) < 5
        )
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "total_units": total_units,
                "blank_title_count": blank_title_count,
                "blank_title_rate": _rate(blank_title_count, total_units),
                "duplicate_title_count": duplicate_title_count,
                "duplicate_title_rate": _rate(duplicate_title_count, total_units),
                "title_content_duplicate_count": title_content_duplicate_count,
                "title_content_duplicate_rate": _rate(title_content_duplicate_count, total_units),
                "very_short_title_count": very_short_title_count,
                "very_short_title_rate": _rate(very_short_title_count, total_units),
            }
        )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_source_type(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_entity_type) or "Unknown"


def _normalized_title(value: object) -> str:
    return _inline_text(value).casefold()


def _rate(count: int, total: int) -> str:
    return f"{(count * 100 / total if total else 0):.2f}"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
