"""Markdown export for unit-level evidence audit reports."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_evidence_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report of evidence by unit."""
    unit_list = sorted(list(units), key=_unit_sort_key)
    edge_list = list(edges)
    markdown = _render(unit_list, edge_list)

    if path is None:
        return markdown

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "edge_count": len(edge_list),
        "evidence_edge_count": _evidence_edge_count(unit_list, edge_list),
        "bytes_written": output_path.stat().st_size,
    }


def _render(units: list[KnowledgeUnit], edges: list[KnowledgeEdge]) -> str:
    units_by_id = {_unit_id(unit): unit for unit in units}
    evidence = _evidence_by_unit(edges, units_by_id)

    lines = ["# Unit Evidence", ""]
    if not units:
        lines.extend(["_No units exported._", ""])
        return "\n".join(lines).rstrip() + "\n"

    for unit in units:
        unit_id = _unit_id(unit)
        lines.extend(
            [
                f"## {_markdown_text(_unit_title(unit))} (`{_code_text(unit_id)}`)",
                "",
                f"- Unit id: `{_code_text(unit_id)}`",
                f"- Unit name: {_markdown_text(_unit_title(unit))}",
                f"- Source: {_markdown_text(_unit_source_name(unit))} (`{_code_text(_unit_source_id(unit))}`)",
                "",
                "### Evidence",
                "",
            ]
        )

        unit_edges = evidence.get(unit_id, [])
        if not unit_edges:
            lines.append("_No evidence._")
        else:
            for edge in sorted(unit_edges, key=lambda item: _evidence_sort_key(item, units_by_id)):
                lines.append(_evidence_line(unit_id, edge, units_by_id))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _evidence_by_unit(
    edges: Iterable[KnowledgeEdge],
    units_by_id: dict[str, KnowledgeUnit],
) -> dict[str, list[KnowledgeEdge]]:
    grouped: dict[str, list[KnowledgeEdge]] = defaultdict(list)
    for edge in edges:
        from_id = _inline_text(edge.from_unit_id)
        to_id = _inline_text(edge.to_unit_id)
        if from_id in units_by_id:
            grouped[from_id].append(edge)
        if to_id in units_by_id and to_id != from_id:
            grouped[to_id].append(edge)
    return dict(grouped)


def _evidence_edge_count(units: list[KnowledgeUnit], edges: list[KnowledgeEdge]) -> int:
    unit_ids = {_unit_id(unit) for unit in units}
    return sum(
        1
        for edge in edges
        if _inline_text(edge.from_unit_id) in unit_ids or _inline_text(edge.to_unit_id) in unit_ids
    )


def _evidence_line(
    unit_id: str,
    edge: KnowledgeEdge,
    units_by_id: dict[str, KnowledgeUnit],
) -> str:
    from_id = _inline_text(edge.from_unit_id)
    to_id = _inline_text(edge.to_unit_id)
    if unit_id == from_id:
        direction = "outgoing"
        support_id = to_id
    else:
        direction = "incoming"
        support_id = from_id

    support_unit = units_by_id.get(support_id)
    parts = [
        f"`{_code_text(_field_value(edge.relation) or 'Unknown')}`",
        direction,
        f"source: {_source_ref(support_unit, support_id)}",
    ]

    confidence = _confidence_text(_edge_confidence(edge))
    if confidence:
        parts.append(f"confidence: {confidence}")

    evidence = _edge_evidence_text(edge)
    if evidence:
        parts.append(f"evidence: {evidence}")

    edge_id = _inline_text(edge.id)
    if edge_id:
        parts.append(f"edge: `{_code_text(edge_id)}`")

    return f"- {'; '.join(parts)}"


def _source_ref(unit: KnowledgeUnit | None, fallback_id: str) -> str:
    if unit is None:
        return f"Unknown (`{_code_text(fallback_id or 'Unknown')}`)"
    return f"{_markdown_text(_unit_source_name(unit))} (`{_code_text(_unit_source_id(unit))}`)"


def _unit_source_name(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    for value in (
        metadata.get("source_name"),
        metadata.get("source_title"),
        metadata.get("site_name"),
        unit.source_project,
    ):
        text = _field_value(value)
        if text:
            return text
    return "Unknown"


def _unit_source_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_id) or "Unknown"


def _unit_title(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    for value in (
        unit.title,
        metadata.get("title"),
        metadata.get("label"),
        metadata.get("name"),
        unit.id,
        unit.source_id,
    ):
        text = _inline_text(value)
        if text:
            return text
    return "Untitled"


def _edge_evidence_text(edge: KnowledgeEdge) -> str:
    metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
    label = _first_metadata_text(metadata, "evidence_label", "label", "title", "name")
    url = _first_metadata_text(metadata, "evidence_url", "url", "source_url")
    if label and url:
        return f"{_markdown_text(label)} ({_markdown_text(url)})"
    if label:
        return _markdown_text(label)
    if url:
        return _markdown_text(url)
    return ""


def _first_metadata_text(metadata: dict, *keys: str) -> str:
    for key in keys:
        text = _inline_text(metadata.get(key))
        if text:
            return text
    return ""


def _edge_confidence(edge: KnowledgeEdge) -> object:
    value = getattr(edge, "confidence", None)
    if value is not None:
        return value
    metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
    return metadata.get("confidence")


def _confidence_text(value: object) -> str:
    if not _is_number(value):
        return ""
    return f"{float(value):.2f}"


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _evidence_sort_key(
    edge: KnowledgeEdge,
    units_by_id: dict[str, KnowledgeUnit],
) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str]]:
    from_unit = units_by_id.get(_inline_text(edge.from_unit_id))
    to_unit = units_by_id.get(_inline_text(edge.to_unit_id))
    from_name = _unit_title(from_unit) if from_unit else _inline_text(edge.from_unit_id)
    to_name = _unit_title(to_unit) if to_unit else _inline_text(edge.to_unit_id)
    return (
        _sort_key(_field_value(edge.relation) or "Unknown"),
        _sort_key(from_name),
        _sort_key(to_name),
        _sort_key(edge.id),
    )


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id or unit.source_id)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_title(unit)), _sort_key(_unit_id(unit)))


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


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
