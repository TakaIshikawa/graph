"""Markdown export for graph units without edges."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_GROUP_BY_VALUES = {"source", "type", "none"}
_WHITESPACE_RE = re.compile(r"\s+")


def export_orphan_units_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    group_by: str = "source",
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report of units with no graph edges."""
    if group_by not in _GROUP_BY_VALUES:
        raise ValueError("group_by must be one of: none, source, type")

    unit_list = list(units)
    connected_ids = {
        unit_id
        for edge in edges
        for unit_id in (_inline_text(edge.from_unit_id), _inline_text(edge.to_unit_id))
        if unit_id
    }
    orphans = sorted(
        [unit for unit in unit_list if _unit_id(unit) not in connected_ids],
        key=_unit_sort_key,
    )
    text = _render_report(orphans, group_by=group_by)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(unit_list),
        "orphans_exported": len(orphans),
        "group_by": group_by,
        "bytes_written": output_path.stat().st_size,
    }


def _render_report(units: list[KnowledgeUnit], *, group_by: str) -> str:
    lines = ["# Orphan Units", ""]
    if not units:
        lines.extend(["_No orphan units found._", ""])
        return "\n".join(lines)

    if group_by == "none":
        lines.extend(_unit_lines(units))
    else:
        groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
        for unit in units:
            groups[_group_label(unit, group_by)].append(unit)
        for label in sorted(groups, key=_sort_key):
            lines.extend([f"## {_markdown_text(label)}", ""])
            lines.extend(_unit_lines(groups[label]))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _unit_lines(units: list[KnowledgeUnit]) -> list[str]:
    lines: list[str] = []
    for unit in sorted(units, key=_unit_sort_key):
        lines.extend(
            [
                f"- **{_markdown_text(_unit_title(unit))}**",
                f"  - ID: `{_code_text(_unit_id(unit))}`",
                f"  - Source: {_markdown_text(_unit_source(unit))}",
                f"  - Type: `{_code_text(_unit_type(unit))}`",
                f"  - Tags: {_markdown_text(_tags_text(unit))}",
                f"  - Preview: {_markdown_text(_preview(unit))}",
            ]
        )
    return lines


def _group_label(unit: KnowledgeUnit, group_by: str) -> str:
    if group_by == "type":
        return _unit_type(unit) or "Unknown"
    return _unit_source(unit) or "Unknown"


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id or unit.source_id)


def _unit_title(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata or {}
    for value in (unit.title, metadata.get("title"), metadata.get("name"), metadata.get("label"), unit.source_id, unit.id):
        text = _inline_text(value)
        if text:
            return text
    return "Untitled"


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or _field_value(unit.content_type) or "Unknown"


def _tags_text(unit: KnowledgeUnit) -> str:
    tags = sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)
    return ", ".join(tags) if tags else "None"


def _preview(unit: KnowledgeUnit, *, max_length: int = 120) -> str:
    metadata = unit.metadata or {}
    text = _inline_text(unit.content) or _inline_text(metadata.get("description")) or _inline_text(metadata.get("summary"))
    if not text:
        return "None"
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    title = _unit_title(unit)
    return (_unit_source(unit).casefold(), _unit_type(unit).casefold(), title.casefold(), _unit_id(unit))


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
