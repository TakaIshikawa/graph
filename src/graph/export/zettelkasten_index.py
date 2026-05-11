"""Zettelkasten-style Markdown index export helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, overload

from graph.types.models import KnowledgeEdge, KnowledgeUnit


@overload
def export_zettelkasten_index_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: None = None,
    *,
    include_orphans: bool = True,
) -> str: ...


@overload
def export_zettelkasten_index_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
    *,
    include_orphans: bool = True,
) -> dict[str, Any]: ...


def export_zettelkasten_index_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    include_orphans: bool = True,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown navigation index."""
    all_units = list(units)
    all_edges = list(edges)
    exported_units = all_units if isinstance(units, Sequence) else sorted(all_units, key=_unit_sort_key)
    by_id = {unit.id: unit for unit in exported_units}
    incoming: dict[str, list[KnowledgeEdge]] = defaultdict(list)
    outgoing: dict[str, list[KnowledgeEdge]] = defaultdict(list)
    for edge in sorted(all_edges, key=_edge_sort_key):
        incoming[edge.to_unit_id].append(edge)
        outgoing[edge.from_unit_id].append(edge)

    lines = ["# Zettelkasten Index", ""]
    lines.extend(_tag_sections(exported_units, incoming, outgoing, by_id))
    lines.extend(_source_sections(exported_units, incoming, outgoing, by_id))
    if include_orphans:
        orphan_units = [
            unit for unit in exported_units if not incoming.get(unit.id) and not outgoing.get(unit.id)
        ]
        if orphan_units:
            lines.extend(["## Orphans", ""])
            lines.extend(_unit_lines(orphan_units, incoming, outgoing, by_id))

    text = "\n".join(lines).rstrip() + "\n"
    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(exported_units),
        "edges_scanned": len(all_edges),
        "bytes_written": output_path.stat().st_size,
    }


def _tag_sections(
    units: Sequence[KnowledgeUnit],
    incoming: dict[str, list[KnowledgeEdge]],
    outgoing: dict[str, list[KnowledgeEdge]],
    by_id: dict[str, KnowledgeUnit],
) -> list[str]:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        for tag in sorted(_clean_text(tag) for tag in unit.tags if _clean_text(tag)):
            grouped[tag].append(unit)
    lines = ["## By Tag", ""]
    for tag in sorted(grouped):
        lines.extend([f"### {_heading_text(tag)}", ""])
        lines.extend(_unit_lines(sorted(grouped[tag], key=_unit_sort_key), incoming, outgoing, by_id))
    return lines


def _source_sections(
    units: Sequence[KnowledgeUnit],
    incoming: dict[str, list[KnowledgeEdge]],
    outgoing: dict[str, list[KnowledgeEdge]],
    by_id: dict[str, KnowledgeUnit],
) -> list[str]:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        grouped[str(getattr(unit.source_project, "value", unit.source_project) or "")].append(unit)
    lines = ["## By Source Project", ""]
    for source_project in sorted(grouped):
        lines.extend([f"### {_heading_text(source_project or 'unknown')}", ""])
        lines.extend(_unit_lines(sorted(grouped[source_project], key=_unit_sort_key), incoming, outgoing, by_id))
    return lines


def _unit_lines(
    units: Sequence[KnowledgeUnit],
    incoming: dict[str, list[KnowledgeEdge]],
    outgoing: dict[str, list[KnowledgeEdge]],
    by_id: dict[str, KnowledgeUnit],
) -> list[str]:
    lines: list[str] = []
    for unit in units:
        backlinks = _linked_units(incoming.get(unit.id, []), by_id, direction="from")
        outgoing_links = _linked_units(outgoing.get(unit.id, []), by_id, direction="to")
        lines.append(f"- `{_inline(unit.id)}` {_link_text(unit.title)}")
        lines.append(f"  Preview: {_inline(_preview(unit.content))}")
        lines.append(f"  Backlinks: {backlinks or 'none'}")
        lines.append(f"  Outgoing: {outgoing_links or 'none'}")
    if lines:
        lines.append("")
    return lines


def _linked_units(edges: Sequence[KnowledgeEdge], by_id: dict[str, KnowledgeUnit], *, direction: str) -> str:
    ids = [edge.from_unit_id if direction == "from" else edge.to_unit_id for edge in edges]
    parts = []
    for unit_id in sorted(ids):
        unit = by_id.get(unit_id)
        label = unit.title if unit else unit_id
        parts.append(f"`{_inline(unit_id)}` {_link_text(label)}")
    return ", ".join(parts)


def _preview(content: object, *, limit: int = 96) -> str:
    text = _clean_text(content)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _heading_text(value: object) -> str:
    return _clean_text(value).replace("#", "\\#")


def _link_text(value: object) -> str:
    return _clean_text(value).replace("[", "\\[").replace("]", "\\]")


def _inline(value: object) -> str:
    return _clean_text(value).replace("`", "\\`")


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        str(getattr(unit.source_project, "value", unit.source_project) or ""),
        str(unit.source_id or ""),
        str(unit.title or ""),
    )


def _edge_sort_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (
        str(edge.from_unit_id or ""),
        str(edge.to_unit_id or ""),
        str(getattr(edge.relation, "value", edge.relation) or ""),
        str(edge.id or ""),
    )
