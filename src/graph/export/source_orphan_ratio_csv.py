"""CSV export for source/type orphan unit ratios."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "unit_count",
    "connected_unit_count",
    "orphan_unit_count",
    "orphan_ratio",
    "incoming_edge_count",
    "outgoing_edge_count",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_orphan_ratio_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge] | None = None,
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write connectedness summaries by source project and entity type."""
    unit_list = list(units)
    edge_list = list(edges or [])
    rows = _orphan_rows(unit_list, edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "edge_count": len(edge_list),
        "source_type_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _orphan_rows(units: list[KnowledgeUnit], edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    units_by_id = {_field_value(unit.id): unit for unit in units if _field_value(getattr(unit, "id", None))}
    connected: set[str] = set()
    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, int] = defaultdict(int)

    for edge in edges:
        from_id = _field_value(getattr(edge, "from_unit_id", None))
        to_id = _field_value(getattr(edge, "to_unit_id", None))
        if from_id in units_by_id and to_id in units_by_id:
            connected.add(from_id)
            connected.add(to_id)
            outgoing[from_id] += 1
            incoming[to_id] += 1

    groups: dict[tuple[str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[
            (
                _field_value(getattr(unit, "source_project", None)) or "Unknown",
                _field_value(getattr(unit, "source_entity_type", None)) or "Unknown",
            )
        ].append(unit)

    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type), group_units in sorted(
        groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))
    ):
        unit_ids = [_field_value(unit.id) for unit in group_units]
        connected_count = sum(1 for unit_id in unit_ids if unit_id in connected)
        orphan_count = len(group_units) - connected_count
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "unit_count": len(group_units),
                "connected_unit_count": connected_count,
                "orphan_unit_count": orphan_count,
                "orphan_ratio": _format_ratio(orphan_count, len(group_units)),
                "incoming_edge_count": sum(incoming[unit_id] for unit_id in unit_ids),
                "outgoing_edge_count": sum(outgoing[unit_id] for unit_id in unit_ids),
            }
        )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00"
    return f"{numerator / denominator:.2f}"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
