"""Markdown export for collection units grouped by tag."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_UNTAGGED = "_untagged"
_WHITESPACE_RE = re.compile(r"\s+")


def export_collection_tag_index_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    title: str = "Collection Tag Index",
    include_untagged: bool = False,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown index of collection units by tag."""
    unit_list = list(units)
    groups = _tag_groups(unit_list, include_untagged=include_untagged)
    text = _render_index(groups, title=_inline_text(title) or "Collection Tag Index")

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "tag_count": len(groups),
        "bytes_written": output_path.stat().st_size,
    }


def _tag_groups(
    units: list[KnowledgeUnit],
    *,
    include_untagged: bool,
) -> dict[str, list[KnowledgeUnit]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        tags = sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)
        if tags:
            for tag in tags:
                groups[tag].append(unit)
        elif include_untagged:
            groups[_UNTAGGED].append(unit)

    return {
        tag: sorted(tag_units, key=_unit_sort_key)
        for tag, tag_units in sorted(groups.items(), key=lambda item: _sort_key(item[0]))
    }


def _render_index(groups: dict[str, list[KnowledgeUnit]], *, title: str) -> str:
    lines = [
        f"# {_inline_markdown(title)}",
        "",
        "## Summary",
        "",
        "| Tag | Units |",
        "| --- | ---: |",
    ]

    if groups:
        for tag, units in groups.items():
            lines.append(f"| {_markdown_cell(tag)} | {len(units)} |")
    else:
        lines.append("| _None_ | 0 |")

    for tag, units in groups.items():
        lines.extend(
            [
                "",
                f"## {_inline_markdown(tag)}",
                "",
                "| ID | Title | Source | Type | Updated |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for unit in units:
            lines.append(
                "| "
                f"{_markdown_cell(_unit_id(unit))} | "
                f"{_markdown_cell(_unit_title(unit))} | "
                f"{_markdown_cell(_unit_source(unit))} | "
                f"{_markdown_cell(_unit_type(unit))} | "
                f"{_markdown_cell(_datetime_text(unit.updated_at))} |"
            )

    return "\n".join(lines).rstrip() + "\n"


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id or unit.source_id)


def _unit_title(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata or {}
    for value in (
        unit.title,
        metadata.get("title"),
        metadata.get("name"),
        metadata.get("label"),
        unit.source_id,
        unit.id,
    ):
        text = _inline_text(value)
        if text:
            return text
    return "Untitled"


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or _field_value(unit.content_type) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (_unit_title(unit).casefold(), _unit_source(unit).casefold(), _unit_id(unit))


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
