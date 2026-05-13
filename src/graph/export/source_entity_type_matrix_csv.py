"""CSV matrix export for source entity type counts across source projects."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_source_entity_type_matrix_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_count: int = 1,
) -> str | dict[str, Any]:
    """Return or write a matrix of source entity type counts by source project."""
    _validate_min_count(min_count)

    unit_list = list(units)
    rows, source_projects = _matrix_rows(unit_list, min_count=min_count)
    text = _render_csv(rows, ["source_entity_type", *source_projects, "total"])

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_project_count": len(source_projects),
        "source_entity_type_count": len(rows),
        "rows_exported": len(rows),
        "min_count": min_count,
        "bytes_written": output_path.stat().st_size,
    }


def _matrix_rows(
    units: list[KnowledgeUnit],
    *,
    min_count: int,
) -> tuple[list[dict[str, str | int]], list[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen_ids: set[str] = set()

    for index, unit in enumerate(units):
        unit_id = _inline_text(unit.id)
        dedupe_key = unit_id or f"__row_{index}"
        if unit_id and dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)

        entity_type = _unit_type(unit)
        source_project = _unit_source(unit)
        counts[entity_type][source_project] += 1

    source_projects = sorted(
        {source_project for project_counts in counts.values() for source_project in project_counts},
        key=_sort_key,
    )
    rows: list[dict[str, str | int]] = []

    for entity_type in sorted(counts, key=_sort_key):
        total = sum(counts[entity_type].values())
        if total < min_count:
            continue
        row: dict[str, str | int] = {"source_entity_type": entity_type}
        for source_project in source_projects:
            row[source_project] = counts[entity_type][source_project]
        row["total"] = total
        rows.append(row)

    return rows, source_projects


def _render_csv(rows: list[dict[str, str | int]], fieldnames: list[str]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_min_count(min_count: int) -> None:
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
