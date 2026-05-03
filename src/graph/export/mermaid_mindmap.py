"""Mermaid mindmap export helpers for tagged knowledge units."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from urllib.parse import urlsplit

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")
_UNTAGGED_LABEL = "Untagged"
_UNCOLLECTED_LABEL = "Uncollected"
_SOURCE_URL_KEYS = (
    "source_url",
    "canonical_url",
    "external_url",
    "permalink",
    "url",
    "href",
    "link",
)
_SOURCE_COLLECTION_KEYS = (
    "source_collection",
    "source_collection_name",
    "collection",
    "collection_name",
    "collections",
)
_GROUP_BY_TAG = "tag"
_GROUP_BY_SOURCE_COLLECTION = "source_collection"


def export_units_to_mermaid_mindmap(
    units: Iterable[KnowledgeUnit],
    *,
    max_units_per_tag: int | None = None,
    include_untagged: bool = False,
    group_by: str = _GROUP_BY_TAG,
    include_source_links: bool = False,
    root_label: str = "Knowledge Units",
) -> str:
    """Return a deterministic Mermaid mindmap grouped by tag or source collection."""
    if max_units_per_tag is not None and (
        not isinstance(max_units_per_tag, int)
        or isinstance(max_units_per_tag, bool)
        or max_units_per_tag < 0
    ):
        raise ValueError("max_units_per_tag must be a non-negative integer or None")
    if not isinstance(include_untagged, bool):
        raise ValueError("include_untagged must be a boolean")
    if not isinstance(include_source_links, bool):
        raise ValueError("include_source_links must be a boolean")

    normalized_group_by = _normalize_group_by(group_by)

    groups = _unit_groups(
        list(units),
        include_untagged=include_untagged,
        group_by=normalized_group_by,
    )
    group_prefix = "tag" if normalized_group_by == _GROUP_BY_TAG else "source"
    lines = [
        "mindmap",
        f'  root["{_mermaid_label(root_label)}"]',
    ]
    click_lines: list[str] = []

    for group_index, (group, grouped_units) in enumerate(groups):
        group_node_id = f"{group_prefix}_{group_index}"
        lines.append(f'    {group_node_id}["{_mermaid_label(group)}"]')
        for unit_index, unit in enumerate(grouped_units[:max_units_per_tag]):
            unit_node_id = f"{group_node_id}_unit_{unit_index}"
            lines.append(
                f'      {unit_node_id}["{_mermaid_label(_unit_title(unit))}"]'
            )
            if include_source_links:
                url = _unit_source_url(unit)
                if url:
                    click_lines.append(
                        f'click {unit_node_id} "{_mermaid_click_url(url)}" "Open source"'
                    )

    lines.extend(click_lines)

    return "\n".join(lines).rstrip() + "\n"


def _unit_groups(
    units: list[KnowledgeUnit],
    *,
    include_untagged: bool,
    group_by: str,
) -> list[tuple[str, list[KnowledgeUnit]]]:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in sorted(units, key=_unit_sort_key):
        groups = _unit_tags(unit) if group_by == _GROUP_BY_TAG else _unit_source_collections(unit)
        if not groups and include_untagged:
            groups = [_UNTAGGED_LABEL if group_by == _GROUP_BY_TAG else _UNCOLLECTED_LABEL]
        for group in groups:
            grouped[group].append(unit)

    return [
        (group, sorted(grouped_units, key=_unit_sort_key))
        for group, grouped_units in sorted(
            grouped.items(),
            key=lambda item: _text_sort_key(item[0]),
        )
    ]


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted(
        {_inline_text(tag) for tag in unit.tags if _inline_text(tag)},
        key=_text_sort_key,
    )


def _unit_source_collections(unit: KnowledgeUnit) -> list[str]:
    groups: set[str] = set()
    for key in _SOURCE_COLLECTION_KEYS:
        groups.update(_metadata_text_values(unit.metadata.get(key)))
    if not groups:
        source_project = _field_value(unit.source_project)
        source_entity_type = _inline_text(unit.source_entity_type)
        if source_project and source_entity_type:
            groups.add(f"{source_project}/{source_entity_type}")
        elif source_project:
            groups.add(source_project)
        elif source_entity_type:
            groups.add(source_entity_type)
    return sorted(groups, key=_text_sort_key)


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


def _metadata_text_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        name = _inline_text(value.get("name") or value.get("title") or value.get("label"))
        return [name] if name else []
    if isinstance(value, list | tuple | set):
        values: list[str] = []
        for item in value:
            values.extend(_metadata_text_values(item))
        return values
    text = _inline_text(value)
    return [text] if text else []


def _unit_source_url(unit: KnowledgeUnit) -> str:
    for key in _SOURCE_URL_KEYS:
        for value in _metadata_text_values(unit.metadata.get(key)):
            url = _url_text(value)
            if url:
                return url
    source_id = _inline_text(unit.source_id)
    if source_id.startswith("url:"):
        source_id = source_id[4:]
    return _url_text(source_id)


def _url_text(value: object) -> str:
    text = _inline_text(value)
    parsed = urlsplit(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return text
    return ""


def _normalize_group_by(value: object) -> str:
    text = _inline_text(value).casefold().replace("-", "_")
    if text in {"tag", "tags"}:
        return _GROUP_BY_TAG
    if text in {
        "source",
        "sources",
        "source_collection",
        "source_collections",
        "collection",
        "collections",
    }:
        return _GROUP_BY_SOURCE_COLLECTION
    raise ValueError("group_by must be 'tag' or 'source_collection'")


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _mermaid_click_url(value: object) -> str:
    return _inline_text(value).replace('"', "%22")


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
