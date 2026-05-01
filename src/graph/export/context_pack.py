"""Markdown context pack export helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_ANCHOR_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_context_pack(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
    *,
    title: str | None = None,
    max_chars: int | None = None,
) -> dict:
    """Write selected units and their included edges as prompt-ready Markdown."""
    if max_chars is not None and max_chars < 1:
        raise ValueError("max_chars must be a positive integer")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_units = list(units)
    all_edges = list(edges)
    included_units: list[KnowledgeUnit] = []

    for unit in all_units:
        candidate_units = [*included_units, unit]
        candidate_text, _ = _render_context_pack(
            candidate_units,
            all_units,
            all_edges,
            title=title,
            max_chars=max_chars,
        )
        if max_chars is not None and len(candidate_text) > max_chars:
            break
        included_units.append(unit)

    text, stats = _render_context_pack(
        included_units,
        all_units,
        all_edges,
        title=title,
        max_chars=max_chars,
    )
    output_path.write_text(text, encoding="utf-8")

    return {"path": str(output_path), **stats}


def _render_context_pack(
    included_units: list[KnowledgeUnit],
    all_units: list[KnowledgeUnit],
    all_edges: list[KnowledgeEdge],
    *,
    title: str | None,
    max_chars: int | None,
) -> tuple[str, dict]:
    included_ids = {unit.id for unit in included_units}
    included_edges = sorted(
        (
            edge
            for edge in all_edges
            if edge.from_unit_id in included_ids and edge.to_unit_id in included_ids
        ),
        key=lambda edge: (
            _field_value(edge.relation),
            edge.from_unit_id,
            edge.to_unit_id,
            edge.id,
        ),
    )
    skipped_edges = len(all_edges) - len(included_edges)
    stats = {
        "units_scanned": len(all_units),
        "units_included": len(included_units),
        "units_skipped": len(all_units) - len(included_units),
        "edges_scanned": len(all_edges),
        "edges_included": len(included_edges),
        "edges_skipped": skipped_edges,
    }

    lines = _summary_lines(title, stats, max_chars)
    lines.extend(_source_index_lines(included_units))
    lines.extend(_unit_section_lines(included_units))
    lines.extend(_edge_section_lines(included_edges))

    text = "\n".join(lines).rstrip() + "\n"
    stats["chars_written"] = len(text)
    return text, stats


def _summary_lines(title: str | None, stats: dict, max_chars: int | None) -> list[str]:
    pack_title = _inline_text(title or "Graph Context Pack") or "Graph Context Pack"
    lines = [
        "---",
        f"title: {_yaml_scalar(pack_title)}",
        f"units_scanned: {stats['units_scanned']}",
        f"units_included: {stats['units_included']}",
        f"units_skipped: {stats['units_skipped']}",
        f"edges_scanned: {stats['edges_scanned']}",
        f"edges_included: {stats['edges_included']}",
        f"edges_skipped: {stats['edges_skipped']}",
    ]
    if max_chars is not None:
        lines.append(f"max_chars: {max_chars}")
    lines.extend(["---", "", f"# {_heading_text(pack_title)}", ""])
    return lines


def _source_index_lines(units: list[KnowledgeUnit]) -> list[str]:
    lines = ["## Source Index", ""]
    if not units:
        return [*lines, "_No units included._", ""]

    for index, unit in enumerate(units, start=1):
        title = _link_text(unit.title or "Untitled")
        source = _inline_text(f"{_field_value(unit.source_project)}/{unit.source_entity_type}")
        source_id = _inline_text(unit.source_id)
        lines.append(
            f"{index}. [{title}](#{_unit_anchor(unit)}) - `{unit.id}` - {source} - `{source_id}`"
        )
    lines.append("")
    return lines


def _unit_section_lines(units: list[KnowledgeUnit]) -> list[str]:
    lines = ["## Units", ""]
    if not units:
        return [*lines, "_No units included._", ""]

    for unit in units:
        lines.extend(
            [
                f'<a id="{_unit_anchor(unit)}"></a>',
                "",
                f"### {_heading_text(unit.title or 'Untitled')}",
                "",
                f"- ID: `{unit.id}`",
                f"- Source: {_inline_text(_field_value(unit.source_project))}/{_inline_text(unit.source_entity_type)}",
                f"- Source ID: `{_inline_text(unit.source_id)}`",
                f"- Type: `{_inline_text(_field_value(unit.content_type))}`",
                f"- Tags: {_tags_text(unit.tags)}",
                "",
                "Content:",
                "",
                _fenced_text(_content_snippet(unit.content)),
                "",
            ]
        )
    return lines


def _edge_section_lines(edges: list[KnowledgeEdge]) -> list[str]:
    lines = ["## Included Edges", ""]
    if not edges:
        return [*lines, "_No edges connect the included units._", ""]

    for edge in edges:
        relation = _inline_text(_field_value(edge.relation))
        source = _inline_text(_field_value(edge.source))
        lines.append(
            f"- `{edge.from_unit_id}` --`{relation}`--> `{edge.to_unit_id}` "
            f"(id: `{edge.id}`, weight: {edge.weight:g}, source: `{source}`)"
        )
    lines.append("")
    return lines


def _field_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _content_snippet(value: object, *, max_length: int = 1200) -> str:
    text = _WHITESPACE_RE.sub(" ", str(value or "")).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


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


def _tags_text(tags: list[str]) -> str:
    if not tags:
        return "_None._"
    return ", ".join(f"`{_inline_text(tag)}`" for tag in tags)


def _unit_anchor(unit: KnowledgeUnit) -> str:
    text = _ANCHOR_RE.sub("-", f"unit-{unit.id}").strip("-").lower()
    return text or "unit"


def _fenced_text(value: str) -> str:
    longest_run = max((len(match.group(0)) for match in re.finditer(r"`+", value)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}\n{value}\n{fence}"


def _yaml_scalar(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'
