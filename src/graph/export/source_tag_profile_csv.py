"""CSV export for source/entity tag usage profiles."""

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
    "unit_count",
    "tagged_unit_count",
    "untagged_unit_count",
    "distinct_tag_count",
    "total_tag_assignments",
    "average_tags_per_unit",
    "top_tags",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_tag_profile_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write tag usage summaries grouped by source and entity type."""
    unit_list = list(units)
    rows = _profile_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "profile_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _profile_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[(_field_value(unit.source_project) or "Unknown", _inline_text(unit.source_entity_type) or "Unknown")].append(
            unit
        )

    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type), group_units in sorted(
        groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))
    ):
        tag_counts: Counter[str] = Counter()
        tagged_unit_count = 0
        for unit in sorted(group_units, key=_unit_sort_key):
            tags = _unit_tags(unit)
            if tags:
                tagged_unit_count += 1
            tag_counts.update(tags)

        total_assignments = sum(tag_counts.values())
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "unit_count": len(group_units),
                "tagged_unit_count": tagged_unit_count,
                "untagged_unit_count": len(group_units) - tagged_unit_count,
                "distinct_tag_count": len(tag_counts),
                "total_tag_assignments": total_assignments,
                "average_tags_per_unit": _decimal(total_assignments / len(group_units)),
                "top_tags": _render_top_tags(tag_counts),
            }
        )
    return rows


def _render_top_tags(counts: Counter[str]) -> str:
    return "; ".join(
        f"{tag} ({count})"
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], *_sort_key(item[0])))
    )


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        _field_value(unit.source_project),
        _inline_text(unit.source_entity_type),
        _inline_text(unit.source_id),
        _inline_text(unit.id),
    )


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
