"""Markdown edge adjacency export helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_adjacency_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    include_backlinks: bool = True,
) -> str:
    """Render a deterministic Markdown adjacency report grouped by unit and relation."""
    exported_units = sorted(units, key=_unit_sort_key)
    units_by_id = {unit.id: unit for unit in exported_units}
    valid_edges = _unique_valid_edges(edges, units_by_id)
    outgoing = _group_edges(valid_edges, source_attr="from_unit_id")
    incoming = _group_edges(valid_edges, source_attr="to_unit_id")

    lines = ["# Edge Adjacency", ""]
    if not exported_units:
        lines.extend(["_No units exported._", ""])
    else:
        for unit in exported_units:
            lines.extend(
                [
                    f"## {_markdown_text(_unit_title(unit))} (`{_code_text(unit.id)}`)",
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

    text = "\n".join(lines).rstrip() + "\n"
    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return text


def _unique_valid_edges(
    edges: Iterable[KnowledgeEdge],
    units_by_id: dict[str, KnowledgeUnit],
) -> list[KnowledgeEdge]:
    unique: dict[tuple[str, str, str], KnowledgeEdge] = {}
    for edge in edges:
        if edge.from_unit_id not in units_by_id or edge.to_unit_id not in units_by_id:
            continue
        key = (
            _field_value(edge.relation),
            _inline_text(edge.from_unit_id),
            _inline_text(edge.to_unit_id),
        )
        current = unique.get(key)
        if current is None or _edge_sort_key(edge, units_by_id, "to_unit_id") < _edge_sort_key(
            current, units_by_id, "to_unit_id"
        ):
            unique[key] = edge
    return sorted(unique.values(), key=lambda edge: _edge_sort_key(edge, units_by_id, "to_unit_id"))


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

    lines: list[str] = []
    for relation in sorted(relation_edges, key=_sort_text):
        lines.extend([f"#### `{_code_text(relation)}`", ""])
        for edge in sorted(
            relation_edges[relation],
            key=lambda edge: _edge_sort_key(edge, units_by_id, neighbor_attr),
        ):
            neighbor = units_by_id[str(getattr(edge, neighbor_attr))]
            lines.append(f"- {_unit_ref(neighbor)}")
        lines.append("")
    return lines[:-1]


def _unit_ref(unit: KnowledgeUnit) -> str:
    return f"{_markdown_text(_unit_title(unit))} (`{_code_text(unit.id)}`)"


def _unit_title(unit: KnowledgeUnit) -> str:
    for value in (
        unit.title,
        unit.metadata.get("title"),
        unit.metadata.get("label"),
        unit.metadata.get("name"),
        unit.id,
        unit.source_id,
    ):
        text = _inline_text(value)
        if text:
            return text
    return "Untitled"


def _edge_sort_key(
    edge: KnowledgeEdge,
    units_by_id: dict[str, KnowledgeUnit],
    neighbor_attr: str,
) -> tuple[str, str, str, str, str]:
    neighbor_id = str(getattr(edge, neighbor_attr))
    return (
        *_unit_sort_key(units_by_id[neighbor_id]),
        _field_value(edge.relation),
        _inline_text(edge.id),
    )


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    title = _unit_title(unit)
    return (title.casefold(), title, _inline_text(unit.id))


def _sort_text(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _markdown_text(value: object) -> str:
    return (
        _inline_text(value)
        .replace("\\", r"\\")
        .replace("*", r"\*")
        .replace("_", r"\_")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("`", r"\`")
    )


def _code_text(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("`", r"\`")


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()
