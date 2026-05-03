"""Neo4j Cypher export helpers for graph data."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit


def export_graph_cypher(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
    *,
    batch_label: str = "KnowledgeUnit",
) -> dict:
    """Write deterministic Neo4j Cypher statements for units and edges."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label = _cypher_identifier(batch_label)
    exported_units = sorted(units, key=lambda unit: _text(unit.id))
    exported_edges = sorted(
        edges,
        key=lambda edge: (
            _text(edge.from_unit_id),
            _text(edge.to_unit_id),
            _enum_value(edge.relation),
            _text(edge.id),
        ),
    )

    lines = [
        f"CREATE CONSTRAINT {label}_id_unique IF NOT EXISTS",
        f"FOR (unit:{label}) REQUIRE unit.id IS UNIQUE;",
        "",
    ]
    for unit in exported_units:
        lines.append(f"MERGE (unit:{label} {{id: {_cypher_literal(unit.id)}}})")
        lines.append(f"SET unit += {_cypher_map(_unit_properties(unit))};")
    for edge in exported_edges:
        relation_type = _relationship_type(edge.relation)
        lines.append(f"MATCH (from:{label} {{id: {_cypher_literal(edge.from_unit_id)}}})")
        lines.append(f"MATCH (to:{label} {{id: {_cypher_literal(edge.to_unit_id)}}})")
        lines.append(f"MERGE (from)-[rel:{relation_type} {{id: {_cypher_literal(edge.id)}}}]->(to)")
        lines.append(f"SET rel += {_cypher_map(_edge_properties(edge))};")
    text = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(text, encoding="utf-8")

    return {
        "path": str(output_path),
        "units_exported": len(exported_units),
        "edges_exported": len(exported_edges),
        "bytes_written": output_path.stat().st_size,
        "label": label,
    }


def _unit_properties(unit: KnowledgeUnit) -> dict[str, Any]:
    return {
        "id": _text(unit.id),
        "source_project": _enum_value(unit.source_project),
        "source_id": _text(unit.source_id),
        "source_entity_type": _text(unit.source_entity_type),
        "title": _text(unit.title),
        "content": _text(unit.content),
        "content_type": _enum_value(unit.content_type),
        "tags": _json_value(unit.tags),
        "metadata": _json_value(unit.metadata),
        "confidence": unit.confidence,
        "utility_score": unit.utility_score,
        "created_at": _json_value(unit.created_at),
        "ingested_at": _json_value(unit.ingested_at),
        "updated_at": _json_value(unit.updated_at),
    }


def _edge_properties(edge: KnowledgeEdge) -> dict[str, Any]:
    return {
        "id": _text(edge.id),
        "from_unit_id": _text(edge.from_unit_id),
        "to_unit_id": _text(edge.to_unit_id),
        "relation": _enum_value(edge.relation),
        "weight": edge.weight,
        "source": _enum_value(edge.source),
        "metadata": _json_value(edge.metadata),
        "created_at": _json_value(edge.created_at),
    }


def _cypher_literal(value: Any) -> str:
    value = _json_value(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return _cypher_string(str(value))
        return repr(value)
    if isinstance(value, str):
        return _cypher_string(value)
    if isinstance(value, list):
        return "[" + ", ".join(_cypher_literal(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return _cypher_map(value)
    return _cypher_string(str(value))


def _cypher_map(values: Mapping[Any, Any]) -> str:
    items = [
        f"{_map_key(key)}: {_cypher_literal(value)}"
        for key, value in sorted(values.items(), key=_item_key)
    ]
    return "{" + ", ".join(items) + "}"


def _cypher_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("'", "\\'")
    )
    return f"'{escaped}'"


def _map_key(key: Any) -> str:
    text = str(key)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return text
    escaped = text.replace("`", "``").replace("\r", "\\r").replace("\n", "\\n")
    return f"`{escaped}`"


def _relationship_type(value: object) -> str:
    text = _enum_value(value).upper()
    return _cypher_identifier(text)


def _cypher_identifier(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "KnowledgeUnit"
    if text[0].isdigit():
        text = f"_{text}"
    return text


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=_item_key)}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return str(value)


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])


def _enum_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _text(value: object) -> str:
    return str(value or "")
