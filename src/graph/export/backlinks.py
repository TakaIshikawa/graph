"""Markdown backlinks export helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_backlinks_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
    *,
    include_orphans: bool = True,
) -> dict[str, int | str]:
    """Write a deterministic Markdown report of incoming and outgoing unit backlinks."""
    all_units = sorted(list(units), key=_unit_sort_key)
    all_edges = list(edges)
    units_by_id = {unit.id: unit for unit in all_units}
    valid_edges = _unique_valid_edges(all_edges, units_by_id)
    incoming = _group_edges(valid_edges, unit_attr="to_unit_id")
    outgoing = _group_edges(valid_edges, unit_attr="from_unit_id")

    exported_units = [
        unit
        for unit in all_units
        if include_orphans or incoming.get(unit.id) or outgoing.get(unit.id)
    ]
    text = _render_report(exported_units, incoming, outgoing, units_by_id)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    bytes_written = len(text.encode("utf-8"))

    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(exported_units),
        "edges_scanned": len(all_edges),
        "bytes_written": bytes_written,
    }


def _render_report(
    units: list[KnowledgeUnit],
    incoming: dict[str, dict[str, list[KnowledgeEdge]]],
    outgoing: dict[str, dict[str, list[KnowledgeEdge]]],
    units_by_id: dict[str, KnowledgeUnit],
) -> str:
    lines = ["# Unit Backlinks", ""]

    if not units:
        lines.extend(["_No units exported._", ""])
    else:
        for unit in units:
            lines.extend(
                [
                    f"## {_markdown_text(_unit_title(unit))} (`{_code_text(unit.id)}`)",
                    "",
                    "### Incoming",
                    "",
                ]
            )
            lines.extend(
                _relation_lines(
                    incoming.get(unit.id, {}),
                    units_by_id,
                    neighbor_attr="from_unit_id",
                )
            )
            lines.extend(["", "### Outgoing", ""])
            lines.extend(
                _relation_lines(
                    outgoing.get(unit.id, {}),
                    units_by_id,
                    neighbor_attr="to_unit_id",
                )
            )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


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
        if current is None or _inline_text(edge.id) < _inline_text(current.id):
            unique[key] = edge
    return sorted(
        unique.values(),
        key=lambda edge: _edge_sort_key(edge, units_by_id, "to_unit_id"),
    )


def _group_edges(
    edges: Iterable[KnowledgeEdge],
    *,
    unit_attr: str,
) -> dict[str, dict[str, list[KnowledgeEdge]]]:
    grouped: dict[str, dict[str, list[KnowledgeEdge]]] = defaultdict(lambda: defaultdict(list))
    for edge in edges:
        grouped[str(getattr(edge, unit_attr))][_field_value(edge.relation)].append(edge)
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
