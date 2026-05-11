"""CSV export helpers for source inventory reports."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "source",
    "source_project",
    "unit_count",
    "edge_count",
    "first_timestamp",
    "last_timestamp",
    "types",
    "tags",
]
_UNKNOWN = "unknown"


def export_source_inventory_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge] | None = None,
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write deterministic CSV grouped by source project."""
    unit_list = list(units)
    edge_list = list(edges or [])
    rows = _inventory_rows(unit_list, edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "rows_written": len(rows)}


def _inventory_rows(units: list[KnowledgeUnit], edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        grouped[_source_project(unit)].append(unit)

    edge_counts = _edge_counts(edges, units)
    rows: list[dict[str, str | int]] = []
    for source_project, source_units in grouped.items():
        timestamps = [
            value
            for unit in source_units
            for value in (unit.created_at, unit.updated_at)
            if isinstance(value, datetime)
        ]
        rows.append(
            {
                "source": source_project,
                "source_project": source_project,
                "unit_count": len(source_units),
                "edge_count": edge_counts[source_project],
                "first_timestamp": min(timestamps).isoformat() if timestamps else "",
                "last_timestamp": max(timestamps).isoformat() if timestamps else "",
                "types": _counter_text(Counter(_text(unit.source_entity_type) or _UNKNOWN for unit in source_units)),
                "tags": _counter_text(Counter(tag for unit in source_units for tag in unit.tags if tag)),
            }
        )
    return sorted(rows, key=lambda row: (_sort_text(row["source_project"]), _sort_text(row["source"])))


def _edge_counts(edges: list[KnowledgeEdge], units: list[KnowledgeUnit]) -> Counter[str]:
    unit_sources: dict[str, str] = {}
    for unit in units:
        source = _source_project(unit)
        for key in (unit.id, unit.source_id):
            if key:
                unit_sources[str(key)] = source

    counts: Counter[str] = Counter()
    for edge in edges:
        sources = {
            unit_sources.get(str(edge.from_unit_id), ""),
            unit_sources.get(str(edge.to_unit_id), ""),
        }
        for source in sources:
            if source:
                counts[source] += 1
    return counts


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _counter_text(counter: Counter[str]) -> str:
    return ";".join(
        f"{key}:{count}"
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))
        if key
    )


def _source_project(unit: KnowledgeUnit) -> str:
    return _text(getattr(unit.source_project, "value", unit.source_project)) or _UNKNOWN


def _text(value: object) -> str:
    return str(value or "").strip()


def _sort_text(value: object) -> str:
    return _text(value).casefold()
