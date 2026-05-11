"""Small Graphviz DOT exporter for directed graph data."""

from __future__ import annotations

from collections.abc import Iterable

from graph.types.models import KnowledgeEdge, KnowledgeUnit


def export_graphviz_dot(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    *,
    graph_name: str = "KnowledgeGraph",
    include_edge_labels: bool = True,
) -> str:
    """Return a deterministic Graphviz ``digraph`` DOT representation."""
    sorted_units = sorted(units, key=lambda unit: _text(unit.id))
    sorted_edges = sorted(
        edges,
        key=lambda edge: (
            _text(edge.from_unit_id),
            _text(edge.to_unit_id),
            _text(getattr(edge.relation, "value", edge.relation)),
            _text(edge.id),
        ),
    )

    lines = [f"digraph {_dot_string(graph_name)} {{"]
    for unit in sorted_units:
        lines.append(f"  {_dot_string(unit.id)} [label={_dot_string(unit.title)}];")
    for edge in sorted_edges:
        line = f"  {_dot_string(edge.from_unit_id)} -> {_dot_string(edge.to_unit_id)}"
        if include_edge_labels:
            line += f" [label={_dot_string(getattr(edge.relation, 'value', edge.relation))}]"
        lines.append(line + ";")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dot_string(value: object) -> str:
    text = _text(value)
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def _text(value: object) -> str:
    return str(value or "")
