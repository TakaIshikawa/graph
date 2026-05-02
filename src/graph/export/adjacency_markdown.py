"""Markdown adjacency list export helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_ANCHOR_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_graph_adjacency_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    *,
    include_backlinks: bool = True,
) -> str:
    """Render a deterministic Markdown adjacency list for units and graph edges."""
    exported_units = sorted(units, key=_unit_sort_key)
    units_by_id = {unit.id: unit for unit in exported_units}
    unit_ids = set(units_by_id)
    valid_edges = [
        edge
        for edge in edges
        if edge.from_unit_id in unit_ids and edge.to_unit_id in unit_ids
    ]
    outgoing = _group_edges(valid_edges, source_attr="from_unit_id")
    incoming = _group_edges(valid_edges, source_attr="to_unit_id")

    lines = ["# Graph Adjacency List", ""]
    if not exported_units:
        lines.extend(["_No units exported._", ""])
        return "\n".join(lines).rstrip() + "\n"

    for unit in exported_units:
        lines.extend(
            [
                f'<a id="{_unit_anchor(unit)}"></a>',
                "",
                f"## {_heading_text(unit.title or 'Untitled')}",
                "",
                f"- ID: `{_code_text(unit.id)}`",
                "",
                "### Outgoing",
                "",
            ]
        )
        lines.extend(
            _relation_lines(
                outgoing.get(unit.id, {}),
                units_by_id,
                neighbor_attr="to_unit_id",
            )
        )
        if include_backlinks:
            lines.extend(["", "### Backlinks", ""])
            lines.extend(
                _relation_lines(
                    incoming.get(unit.id, {}),
                    units_by_id,
                    neighbor_attr="from_unit_id",
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _group_edges(
    edges: Iterable[KnowledgeEdge],
    *,
    source_attr: str,
) -> dict[str, dict[str, list[KnowledgeEdge]]]:
    grouped: dict[str, dict[str, list[KnowledgeEdge]]] = defaultdict(lambda: defaultdict(list))
    for edge in edges:
        grouped[str(getattr(edge, source_attr))][_field_value(edge.relation)].append(edge)
    return {unit_id: dict(relations) for unit_id, relations in grouped.items()}


def _relation_lines(
    relation_edges: dict[str, list[KnowledgeEdge]],
    units_by_id: dict[str, KnowledgeUnit],
    *,
    neighbor_attr: str,
) -> list[str]:
    if not relation_edges:
        return ["_None._"]

    lines = []
    for relation in sorted(relation_edges, key=_sort_text):
        links = [
            _unit_link(units_by_id[str(getattr(edge, neighbor_attr))])
            for edge in sorted(
                relation_edges[relation],
                key=lambda edge: _neighbor_sort_key(edge, units_by_id, neighbor_attr),
            )
        ]
        lines.append(f"- `{_code_text(relation)}`: {', '.join(links)}")
    return lines


def _unit_link(unit: KnowledgeUnit) -> str:
    title = _link_text(unit.title or "Untitled")
    return f"[{title}](#{_unit_anchor(unit)})"


def _neighbor_sort_key(
    edge: KnowledgeEdge,
    units_by_id: dict[str, KnowledgeUnit],
    neighbor_attr: str,
) -> tuple[str, str, str, str]:
    neighbor_id = str(getattr(edge, neighbor_attr))
    return (*_unit_sort_key(units_by_id[neighbor_id]), _text(edge.id))


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    title = _text(unit.title or "Untitled")
    return (title.casefold(), title, _text(unit.id))


def _sort_text(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)


def _field_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _heading_text(value: object) -> str:
    text = _inline_text(value).replace("\\", "\\\\").replace("#", r"\#")
    return text or "Untitled"


def _link_text(value: object) -> str:
    text = _inline_text(value)
    return (
        text.replace("\\", r"\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


def _code_text(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("`", r"\`")


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", _text(value)).strip()


def _text(value: object) -> str:
    return str(value or "")


def _unit_anchor(unit: KnowledgeUnit) -> str:
    text = _ANCHOR_RE.sub("-", f"unit-{unit.id}").strip("-").lower()
    return text or "unit"
