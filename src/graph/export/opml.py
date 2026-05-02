"""OPML export helpers for knowledge units."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from xml.etree import ElementTree as ET

from graph.types.models import KnowledgeUnit

UNTAGGED_LABEL = "Untagged"


def export_units_to_opml(
    units: Iterable[KnowledgeUnit],
    *,
    title: str = "Graph Export",
    include_untagged: bool = True,
) -> str:
    """Serialize units as an OPML 2.0 outline grouped by tag."""
    grouped_units = _group_units_by_tag(units, include_untagged=include_untagged)

    root = ET.Element("opml", {"version": "2.0"})
    head = ET.SubElement(root, "head")
    ET.SubElement(head, "title").text = _text(title) or "Graph Export"
    body = ET.SubElement(root, "body")

    for tag in sorted(grouped_units, key=_sort_text):
        tag_node = ET.SubElement(body, "outline", {"text": tag})
        for unit in sorted(grouped_units[tag], key=_unit_sort_key):
            ET.SubElement(tag_node, "outline", _unit_attributes(unit))

    xml = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return f'<?xml version="1.0" encoding="utf-8"?>\n{xml}\n'


def _group_units_by_tag(
    units: Iterable[KnowledgeUnit],
    *,
    include_untagged: bool,
) -> dict[str, list[KnowledgeUnit]]:
    grouped_units: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        tags = sorted({_text(tag) for tag in unit.tags if _text(tag)}, key=_sort_text)
        if not tags:
            if include_untagged:
                grouped_units[UNTAGGED_LABEL].append(unit)
            continue
        for tag in tags:
            grouped_units[tag].append(unit)
    return dict(grouped_units)


def _unit_attributes(unit: KnowledgeUnit) -> dict[str, str]:
    attributes = {
        "text": _text(unit.title) or "Untitled",
        "source_id": _text(unit.source_id),
        "source_project": _field_value(unit.source_project),
        "content_type": _field_value(unit.content_type),
    }
    created_at = _date_text(unit.created_at)
    if created_at:
        attributes["created_at"] = created_at
    updated_at = _date_text(unit.updated_at)
    if updated_at:
        attributes["updated_at"] = updated_at
    return attributes


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (
        _text(unit.title or "Untitled").casefold(),
        _text(unit.title or "Untitled"),
        _text(unit.source_id),
        _text(unit.id),
    )


def _sort_text(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)


def _field_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _date_text(value: object) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return _text(value)


def _text(value: object) -> str:
    return str(value or "").strip()
