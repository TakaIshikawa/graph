"""CSV export for per-unit source diversity."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "unit_name",
    "contributing_source_count",
    "distinct_source_type_count",
    "source_types",
    "evidence_edge_count",
    "top_source",
    "top_source_evidence_count",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_source_diversity_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge] | None = None,
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source diversity evidence grouped by unit."""
    unit_list = list(units)
    edge_list = list(edges or [])
    rows = _diversity_rows(unit_list, edge_list)
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
        "bytes_written": output_path.stat().st_size,
    }


def _diversity_rows(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
) -> list[dict[str, str | int]]:
    units_by_key = _unit_index(units)
    incident_edges = _incident_edges(edges, units_by_key)

    rows: list[dict[str, str | int]] = []
    for unit in sorted(units, key=_unit_sort_key):
        source_counts: Counter[str] = Counter()
        source_types: set[str] = set()

        for edge in incident_edges.get(_unit_id(unit), []):
            other = _other_unit(edge, unit, units_by_key)
            if other is None:
                source_counts["Unknown"] += 1
                continue

            source_counts[_unit_source(other)] += 1
            source_type = _unit_type(other)
            if source_type != "Unknown":
                source_types.add(source_type)

        top_source, top_count = _top_source(source_counts)
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "unit_name": _unit_name(unit),
                "contributing_source_count": len(source_counts),
                "distinct_source_type_count": len(source_types),
                "source_types": _source_types_text(source_types),
                "evidence_edge_count": sum(source_counts.values()),
                "top_source": top_source,
                "top_source_evidence_count": top_count,
            }
        )
    return rows


def _unit_index(units: list[KnowledgeUnit]) -> dict[str, KnowledgeUnit]:
    index: dict[str, KnowledgeUnit] = {}
    for unit in units:
        for key in (_unit_id(unit), _inline_text(unit.source_id)):
            if key:
                index[key] = unit
    return index


def _incident_edges(
    edges: list[KnowledgeEdge],
    units_by_key: dict[str, KnowledgeUnit],
) -> dict[str, list[KnowledgeEdge]]:
    incident: dict[str, list[KnowledgeEdge]] = {}
    for edge in sorted(edges, key=_edge_sort_key):
        endpoint_units = {
            _unit_id(unit)
            for unit in (
                units_by_key.get(_inline_text(edge.from_unit_id)),
                units_by_key.get(_inline_text(edge.to_unit_id)),
            )
            if unit is not None
        }
        for unit_id in endpoint_units:
            incident.setdefault(unit_id, []).append(edge)
    return incident


def _other_unit(
    edge: KnowledgeEdge,
    unit: KnowledgeUnit,
    units_by_key: dict[str, KnowledgeUnit],
) -> KnowledgeUnit | None:
    unit_keys = {_unit_id(unit), _inline_text(unit.source_id)}
    from_id = _inline_text(edge.from_unit_id)
    to_id = _inline_text(edge.to_unit_id)

    if from_id in unit_keys and to_id not in unit_keys:
        return units_by_key.get(to_id)
    if to_id in unit_keys and from_id not in unit_keys:
        return units_by_key.get(from_id)
    return None


def _top_source(source_counts: Counter[str]) -> tuple[str, int]:
    if not source_counts:
        return "", 0
    source, count = min(source_counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))
    return source, count


def _source_types_text(source_types: set[str]) -> str:
    return "; ".join(sorted(source_types, key=_sort_key))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id) or _inline_text(unit.source_id)


def _unit_name(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.title)


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


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_id(unit)), _sort_key(_unit_name(unit)))


def _edge_sort_key(edge: KnowledgeEdge) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    return (_sort_key(edge.from_unit_id), _sort_key(edge.to_unit_id), _sort_key(edge.id))
