"""CSV export for edges connecting different source projects."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "edge_id",
    "relation",
    "from_unit_id",
    "from_source_project",
    "to_unit_id",
    "to_source_project",
    "from_source_entity_type",
    "to_source_entity_type",
    "direction_label",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_cross_source_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    include_unknown: bool = False,
) -> str | dict[str, Any]:
    """Return or write edges whose endpoints belong to different source projects."""
    unit_list = list(units)
    edge_list = list(edges)
    rows = _cross_source_rows(unit_list, edge_list, include_unknown)
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
        "cross_source_edge_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _cross_source_rows(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
    include_unknown: bool,
) -> list[dict[str, str]]:
    units_by_id = {_field_value(unit.id): unit for unit in units if _field_value(getattr(unit, "id", None))}
    rows: list[dict[str, str]] = []
    for edge in edges:
        from_id = _field_value(getattr(edge, "from_unit_id", None))
        to_id = _field_value(getattr(edge, "to_unit_id", None))
        from_unit = units_by_id.get(from_id)
        to_unit = units_by_id.get(to_id)
        if from_unit is None or to_unit is None:
            if not include_unknown:
                continue
        from_source = _unit_field(from_unit, "source_project")
        to_source = _unit_field(to_unit, "source_project")
        if from_source == to_source and "Unknown" not in (from_source, to_source):
            continue
        rows.append(
            {
                "edge_id": _field_value(getattr(edge, "id", None)),
                "relation": _field_value(getattr(edge, "relation", None)),
                "from_unit_id": from_id,
                "from_source_project": from_source,
                "to_unit_id": to_id,
                "to_source_project": to_source,
                "from_source_entity_type": _unit_field(from_unit, "source_entity_type"),
                "to_source_entity_type": _unit_field(to_unit, "source_entity_type"),
                "direction_label": f"{from_source} -> {to_source}",
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["edge_id"]), _sort_key(row["from_unit_id"]), _sort_key(row["to_unit_id"])))


def _unit_field(unit: KnowledgeUnit | None, attr: str) -> str:
    if unit is None:
        return "Unknown"
    return _field_value(getattr(unit, attr, None)) or "Unknown"


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
