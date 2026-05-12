"""CSV export for per-unit edge coverage."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source",
    "type",
    "in_degree",
    "out_degree",
    "total_degree",
    "relations_in",
    "relations_out",
    "is_isolate",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_coverage_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    include_isolates: bool = True,
) -> str | dict[str, Any]:
    """Return or write per-unit inbound and outbound edge coverage as CSV."""
    unit_list = list(units)
    edge_list = list(edges)
    rows = _coverage_rows(unit_list, edge_list, include_isolates=include_isolates)
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
        "rows_exported": len(rows),
        "include_isolates": include_isolates,
        "bytes_written": output_path.stat().st_size,
    }


def _coverage_rows(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
    *,
    include_isolates: bool,
) -> list[dict[str, str | int]]:
    units_by_id = {_unit_id(unit): unit for unit in units}
    inbound: dict[str, list[KnowledgeEdge]] = defaultdict(list)
    outbound: dict[str, list[KnowledgeEdge]] = defaultdict(list)

    for edge in edges:
        from_id = _inline_text(edge.from_unit_id)
        to_id = _inline_text(edge.to_unit_id)
        if from_id in units_by_id:
            outbound[from_id].append(edge)
        if to_id in units_by_id:
            inbound[to_id].append(edge)

    rows: list[dict[str, str | int]] = []
    for unit in units:
        unit_id = _unit_id(unit)
        in_degree = len(inbound[unit_id])
        out_degree = len(outbound[unit_id])
        total_degree = in_degree + out_degree
        if total_degree == 0 and not include_isolates:
            continue
        rows.append(
            {
                "unit_id": unit_id,
                "title": _unit_title(unit),
                "source": _unit_source(unit),
                "type": _unit_type(unit),
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": total_degree,
                "relations_in": _relations_text(inbound[unit_id]),
                "relations_out": _relations_text(outbound[unit_id]),
                "is_isolate": "true" if total_degree == 0 else "false",
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -int(row["total_degree"]),
            _sort_key(row["title"]),
            _sort_key(row["source"]),
            _sort_key(row["unit_id"]),
        ),
    )


def _relations_text(edges: list[KnowledgeEdge]) -> str:
    counts = Counter(_field_value(edge.relation) for edge in edges)
    return "; ".join(
        f"{relation} ({count})"
        for relation, count in sorted(counts.items(), key=lambda item: _sort_key(item[0]))
    )


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id or unit.source_id)


def _unit_title(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.title) or _inline_text((unit.metadata or {}).get("title")) or _unit_id(unit)


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
