"""RDF Turtle export for knowledge units and graph edges."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

GRAPH_NS = "https://example.org/graph/"
UNIT_NS = "https://example.org/graph/unit/"


def export_graph_rdf_turtle(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge] | None = None,
    path: str | Path | None = None,
    *,
    graph_namespace: str = GRAPH_NS,
    unit_namespace: str = UNIT_NS,
) -> str:
    """Return units and optional edges as deterministic Turtle text."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    edge_list = list(edges or [])

    lines = [
        "@prefix kg: <https://example.org/graph/vocab#> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]
    for unit in sorted(unit_list, key=lambda item: _unit_key(item)):
        lines.extend(_unit_lines(unit, unit_namespace=unit_namespace))
        lines.append("")
    for edge in sorted(edge_list, key=_edge_key):
        predicate = f"<{graph_namespace}relation/{_slug(_field_value(edge.relation))}>"
        lines.append(f"{_unit_uri(edge.from_unit_id, unit_namespace)} {predicate} {_unit_uri(edge.to_unit_id, unit_namespace)} .")
    text = "\n".join(lines).rstrip() + "\n"

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return text


def _unit_lines(unit: KnowledgeUnit, *, unit_namespace: str) -> list[str]:
    subject = _unit_uri(unit.id or unit.source_id, unit_namespace)
    predicates: list[tuple[str, str]] = [
        ("a", "kg:KnowledgeUnit"),
        ("dcterms:identifier", _literal(unit.id or unit.source_id)),
        ("dcterms:title", _literal(unit.title)),
        ("kg:content", _literal(unit.content)),
        ("kg:sourceProject", _literal(_field_value(unit.source_project))),
        ("kg:contentType", _literal(_field_value(unit.content_type))),
    ]
    for tag in sorted(str(tag) for tag in unit.tags):
        predicates.append(("kg:tag", _literal(tag)))
    if unit.created_at:
        predicates.append(("dcterms:created", _datetime_literal(unit.created_at)))
    if unit.updated_at:
        predicates.append(("dcterms:modified", _datetime_literal(unit.updated_at)))

    lines = [f"{subject} {predicates[0][0]} {predicates[0][1]} ;"]
    for index, (predicate, obj) in enumerate(predicates[1:], start=1):
        suffix = " ." if index == len(predicates) - 1 else " ;"
        lines.append(f"    {predicate} {obj}{suffix}")
    return lines


def _unit_uri(unit_id: str, unit_namespace: str) -> str:
    return f"<{unit_namespace}{_slug(unit_id)}>"


def _slug(value: object) -> str:
    text = str(value or "unit").strip()
    slug = re.sub(r"[^A-Za-z0-9._~-]+", "-", text).strip("-")
    return slug or "unit"


def _literal(value: object) -> str:
    text = str(value or "")
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"'


def _datetime_literal(value: datetime | date) -> str:
    text = value.isoformat()
    return f'"{text}"^^xsd:dateTime'


def _field_value(value: Any) -> str:
    if isinstance(value, Enum):
        return value.value
    return str(value or "")


def _unit_key(unit: KnowledgeUnit) -> tuple[str, str]:
    return (str(unit.id or ""), str(unit.source_id or ""))


def _edge_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (
        str(edge.from_unit_id or ""),
        str(edge.to_unit_id or ""),
        _field_value(edge.relation),
        str(edge.id or ""),
    )
