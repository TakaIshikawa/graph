"""README-style Markdown export for selected graph collections."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_collection_readme_markdown(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge] = (),
    path: str | Path | None = None,
    *,
    title: str = "Collection README",
    generated_at: str | datetime | None = None,
    include_edges: bool = True,
) -> str | dict[str, Any]:
    """Return or write a deterministic README-style Markdown collection report."""
    unit_list = sorted(list(units), key=_unit_sort_key)
    unit_ids = {_unit_id(unit) for unit in unit_list}
    edge_list = sorted(
        [
            edge
            for edge in edges
            if include_edges
            and _inline_text(edge.from_unit_id) in unit_ids
            and _inline_text(edge.to_unit_id) in unit_ids
        ],
        key=_edge_sort_key,
    )

    text = _render_readme(
        unit_list,
        edge_list,
        title=_inline_text(title) or "Collection README",
        generated_at=generated_at,
    )
    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "edge_count": len(edge_list),
        "bytes_written": output_path.stat().st_size,
    }


def _render_readme(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
    *,
    title: str,
    generated_at: str | datetime | None,
) -> str:
    lines = [
        f"# {_inline_markdown(title)}",
        "",
    ]
    if generated_at is not None:
        lines.extend([f"Generated at: {_inline_markdown(_datetime_text(generated_at))}", ""])

    lines.extend(
        [
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Units | {len(units)} |",
            f"| Edges | {len(edges)} |",
            "",
            "## Sources",
            "",
            "| Source | Units |",
            "| --- | ---: |",
        ]
    )
    source_counts = Counter(_unit_source(unit) for unit in units)
    if source_counts:
        for source, count in sorted(source_counts.items(), key=lambda item: _sort_key(item[0])):
            lines.append(f"| {_markdown_cell(source)} | {count} |")
    else:
        lines.append("| _None_ | 0 |")

    lines.extend(["", "## Tags", "", "| Tag | Units |", "| --- | ---: |"])
    tag_counts = _tag_counts(units)
    if tag_counts:
        for tag, count in sorted(tag_counts.items(), key=lambda item: _sort_key(item[0])):
            lines.append(f"| {_markdown_cell(tag)} | {count} |")
    else:
        lines.append("| _None_ | 0 |")

    lines.extend(
        [
            "",
            "## Units",
            "",
            "| ID | Title | Source | Type | Tags |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    if units:
        for unit in units:
            lines.append(
                "| "
                f"{_markdown_cell(_unit_id(unit))} | "
                f"{_markdown_cell(_unit_title(unit))} | "
                f"{_markdown_cell(_unit_source(unit))} | "
                f"{_markdown_cell(_unit_type(unit))} | "
                f"{_markdown_cell(_unit_tags_text(unit))} |"
            )
    else:
        lines.append("| _None_ | _None_ | _None_ | _None_ | _None_ |")

    if edges:
        lines.extend(
            [
                "",
                "## Edges",
                "",
                "| From | Relation | To | Source | Weight |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for edge in edges:
            lines.append(
                "| "
                f"{_markdown_cell(edge.from_unit_id)} | "
                f"{_markdown_cell(_field_value(edge.relation))} | "
                f"{_markdown_cell(edge.to_unit_id)} | "
                f"{_markdown_cell(_field_value(edge.source))} | "
                f"{edge.weight:g} |"
            )

    return "\n".join(lines).rstrip() + "\n"


def _tag_counts(units: Iterable[KnowledgeUnit]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for unit in units:
        for tag in {_inline_text(tag) for tag in unit.tags if _inline_text(tag)}:
            counts[tag] += 1
    return counts


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


def _unit_tags_text(unit: KnowledgeUnit) -> str:
    tags = sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)
    return ", ".join(tags) if tags else "_None_"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (_unit_title(unit).casefold(), _unit_source(unit).casefold(), _unit_id(unit))


def _edge_sort_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (
        _inline_text(edge.from_unit_id),
        _inline_text(edge.to_unit_id),
        _field_value(edge.relation),
        _inline_text(edge.id),
    )


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _datetime_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return _inline_text(value)


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _inline_markdown(value: object) -> str:
    return (
        _inline_text(value)
        .replace("\\", r"\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")
