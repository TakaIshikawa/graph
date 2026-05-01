"""Anki-compatible TSV export helpers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

Template = str | Callable[[KnowledgeUnit], object]

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_INVALID_RE = re.compile(r"[^0-9A-Za-z_-]+")


def export_units_to_anki_tsv(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
    *,
    front_template: Template | None = None,
    back_template: Template | None = None,
    include_tags: bool = True,
) -> dict:
    """Write units as one TSV row per Anki card.

    Templates may be callables that accept a unit or format strings using unit
    fields such as ``{title}``, ``{content}``, ``{source_id}``, and ``{unit.id}``.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_units = list(units)
    rows: list[list[str]] = []
    skipped_empty = 0

    for unit in all_units:
        front = _field_text(_render_front(unit, front_template))
        back = _field_text(_render_back(unit, back_template))
        if not front or not back:
            skipped_empty += 1
            continue

        row = [front, back]
        if include_tags:
            row.append(_tags_text(unit.tags))
        rows.append(row)

    text = "".join("\t".join(row) + "\n" for row in rows)
    output_path.write_text(text, encoding="utf-8")

    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "cards_exported": len(rows),
        "skipped_empty": skipped_empty,
    }


def _render_front(unit: KnowledgeUnit, template: Template | None) -> object:
    if template is None:
        return unit.title
    return _render_template(unit, template)


def _render_back(unit: KnowledgeUnit, template: Template | None) -> object:
    if template is not None:
        return _render_template(unit, template)

    content = _inline_text(unit.content)
    if not content:
        return ""
    source = (
        f"Source: {_field_value(unit.source_project)}/"
        f"{_inline_text(unit.source_entity_type)}"
        f" ({_inline_text(unit.source_id)})"
    )
    return "\n\n".join(part for part in [content, source] if part)


def _render_template(unit: KnowledgeUnit, template: Template) -> object:
    if callable(template):
        return template(unit)
    return template.format(
        unit=unit,
        id=unit.id,
        title=unit.title,
        content=unit.content,
        content_type=_field_value(unit.content_type),
        source_project=_field_value(unit.source_project),
        source_id=unit.source_id,
        source_entity_type=unit.source_entity_type,
        metadata=unit.metadata,
        tags=unit.tags,
    )


def _field_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _inline_text(value: object) -> str:
    return _field_text(value)


def _field_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _tags_text(tags: list[str]) -> str:
    normalized: set[str] = set()
    for tag in tags:
        normalized_tag = _normalize_tag(tag)
        if normalized_tag:
            normalized.add(normalized_tag)
    return " ".join(sorted(normalized))


def _normalize_tag(value: Any) -> str:
    tag = _TAG_INVALID_RE.sub("_", _field_text(value)).strip("_")
    return tag.lower()
