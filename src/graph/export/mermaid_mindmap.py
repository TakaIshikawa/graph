"""Mermaid mindmap export helpers for tagged knowledge units."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")
_UNTAGGED_LABEL = "Untagged"


def export_units_to_mermaid_mindmap(
    units: Iterable[KnowledgeUnit],
    *,
    max_units_per_tag: int | None = None,
    include_untagged: bool = False,
    root_label: str = "Knowledge Units",
) -> str:
    """Return a deterministic Mermaid mindmap grouped by unit tag."""
    if max_units_per_tag is not None and (
        not isinstance(max_units_per_tag, int)
        or isinstance(max_units_per_tag, bool)
        or max_units_per_tag < 0
    ):
        raise ValueError("max_units_per_tag must be a non-negative integer or None")
    if not isinstance(include_untagged, bool):
        raise ValueError("include_untagged must be a boolean")

    groups = _tag_groups(list(units), include_untagged=include_untagged)
    lines = [
        "mindmap",
        f'  root["{_mermaid_label(root_label)}"]',
    ]

    for tag_index, (tag, tagged_units) in enumerate(groups):
        lines.append(f'    tag_{tag_index}["{_mermaid_label(tag)}"]')
        for unit_index, unit in enumerate(tagged_units[:max_units_per_tag]):
            lines.append(
                f'      tag_{tag_index}_unit_{unit_index}["{_mermaid_label(_unit_title(unit))}"]'
            )

    return "\n".join(lines).rstrip() + "\n"


def _tag_groups(
    units: list[KnowledgeUnit],
    *,
    include_untagged: bool,
) -> list[tuple[str, list[KnowledgeUnit]]]:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in sorted(units, key=_unit_sort_key):
        tags = _unit_tags(unit)
        if not tags and include_untagged:
            tags = [_UNTAGGED_LABEL]
        for tag in tags:
            grouped[tag].append(unit)

    return [
        (tag, sorted(tagged_units, key=_unit_sort_key))
        for tag, tagged_units in sorted(grouped.items(), key=lambda item: _text_sort_key(item[0]))
    ]


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted(
        {_inline_text(tag) for tag in unit.tags if _inline_text(tag)},
        key=_text_sort_key,
    )


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


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    title = _unit_title(unit)
    return (*_text_sort_key(title), _inline_text(unit.id))


def _text_sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _mermaid_label(value: object) -> str:
    return (
        _inline_text(value)
        .replace("&", "&amp;")
        .replace("\\", "&#92;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("(", "&#40;")
        .replace(")", "&#41;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
        .replace("`", "&#96;")
    )
