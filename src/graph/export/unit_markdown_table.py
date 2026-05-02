"""Markdown table export helpers for knowledge units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

DEFAULT_FIELDS = [
    "id",
    "title",
    "source_project",
    "content_type",
    "created_at",
    "tags",
    "content_excerpt",
]

_EXCERPT_LIMIT = 120
_WHITESPACE_RE = re.compile(r"\s+")


def export_units_to_markdown_table(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    fields: Sequence[str] | None = None,
) -> str:
    """Render units as a deterministic GitHub-flavored Markdown table."""
    fieldnames = list(fields) if fields is not None else [*DEFAULT_FIELDS]
    all_units = sorted(list(units), key=_unit_sort_key)

    text = _render_table(all_units, fieldnames)
    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return text


def _render_table(units: list[KnowledgeUnit], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(_markdown_cell(field) for field in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for unit in units:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(_field_value(unit, field)) for field in fields)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _field_value(unit: KnowledgeUnit, field: str) -> Any:
    if field == "content_excerpt":
        return _content_excerpt(unit.content)
    if field == "tags":
        return "; ".join(sorted(_inline_text(tag) for tag in unit.tags))
    return getattr(unit, field, "")


def _content_excerpt(value: object) -> str:
    text = _inline_text(value)
    if len(text) <= _EXCERPT_LIMIT:
        return text
    return text[: _EXCERPT_LIMIT - len("...")].rstrip() + "..."


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        _inline_text(unit.id),
        _inline_text(_value_text(unit.source_project)),
        _inline_text(unit.source_id),
    )


def _markdown_cell(value: object) -> str:
    return _inline_text(_value_text(value)).replace("\\", "\\\\").replace("|", "\\|")


def _value_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()
