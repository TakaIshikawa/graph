"""CSV heatmap export for source project and entity type counts."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_TOTAL_ROW = "__total__"
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_entity_type_heatmap_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a CSV heatmap of unit counts by source and entity type."""
    unit_list = list(units)
    rows, entity_types = _heatmap_rows(unit_list)
    text = _render_csv(rows, ["source_project", *entity_types, "total"])

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_project_count": len(
            {row["source_project"] for row in rows if row["source_project"] != _TOTAL_ROW}
        ),
        "source_entity_type_count": len(entity_types),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _heatmap_rows(units: list[KnowledgeUnit]) -> tuple[list[dict[str, str | int]], list[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    entity_types: set[str] = set()

    for unit in units:
        source_project = _unit_source(unit)
        entity_type = _unit_type(unit)
        counts[source_project][entity_type] += 1
        entity_types.add(entity_type)

    sorted_entity_types = sorted(entity_types, key=_sort_key)
    rows: list[dict[str, str | int]] = []
    totals: Counter[str] = Counter()

    for source_project in sorted(counts, key=_sort_key):
        row: dict[str, str | int] = {"source_project": source_project}
        row_total = 0
        for entity_type in sorted_entity_types:
            count = counts[source_project][entity_type]
            row[entity_type] = count
            totals[entity_type] += count
            row_total += count
        row["total"] = row_total
        rows.append(row)

    total_row: dict[str, str | int] = {"source_project": _TOTAL_ROW}
    grand_total = 0
    for entity_type in sorted_entity_types:
        count = totals[entity_type]
        total_row[entity_type] = count
        grand_total += count
    total_row["total"] = grand_total
    rows.append(total_row)

    return rows, sorted_entity_types


def _render_csv(rows: list[dict[str, str | int]], fieldnames: list[str]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or _field_value(unit.content_type) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
