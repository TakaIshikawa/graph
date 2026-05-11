"""Dublin Core XML export helpers for knowledge units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, overload
from xml.etree import ElementTree as ET

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

DC_NS = "http://purl.org/dc/elements/1.1/"
ET.register_namespace("dc", DC_NS)

_CREATOR_KEYS = ("creator", "creators", "author", "authors")
_DESCRIPTION_KEYS = ("description", "summary", "abstract")
_PUBLISHER_KEYS = ("publisher", "publication", "journal")
_DATE_KEYS = ("date", "published_at", "publication_date", "issued", "year")
_TYPE_KEYS = ("type", "kind", "document_type")
_IDENTIFIER_KEYS = ("doi", "DOI", "url", "source_url", "external_url", "uri")
_SOURCE_KEYS = ("source", "source_title", "container_title")
_LANGUAGE_KEYS = ("language", "lang")
_RELATION_KEYS = ("relation", "related", "references")


@overload
def export_units_to_dublin_core_xml(
    units: Iterable[KnowledgeUnit],
    path: None = None,
) -> str: ...


@overload
def export_units_to_dublin_core_xml(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
) -> dict[str, Any]: ...


def export_units_to_dublin_core_xml(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write deterministic Dublin Core XML records."""
    all_units = list(units)
    exported_units = all_units if isinstance(units, Sequence) else sorted(all_units, key=_unit_sort_key)
    root = ET.Element("records")
    for unit in exported_units:
        root.append(_record(unit))
    text = ET.tostring(root, encoding="unicode", short_empty_elements=True) + "\n"

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(exported_units),
        "bytes_written": output_path.stat().st_size,
    }


def _record(unit: KnowledgeUnit) -> ET.Element:
    record = ET.Element("record")
    _add(record, "title", unit.title)
    for value in _metadata_values(unit.metadata, _CREATOR_KEYS):
        _add(record, "creator", value)
    for tag in sorted(_clean_text(tag) for tag in unit.tags if _clean_text(tag)):
        _add(record, "subject", tag)
    _add(record, "description", _first_text(unit.metadata, _DESCRIPTION_KEYS) or unit.content)
    _add(record, "publisher", _first_text(unit.metadata, _PUBLISHER_KEYS))
    _add(record, "date", _first_text(unit.metadata, _DATE_KEYS) or unit.created_at)
    _add(record, "type", _first_text(unit.metadata, _TYPE_KEYS) or unit.content_type)
    _add(record, "identifier", unit.source_id)
    for value in _metadata_values(unit.metadata, _IDENTIFIER_KEYS):
        _add(record, "identifier", value)
    _add(record, "source", _first_text(unit.metadata, _SOURCE_KEYS) or unit.source_project)
    _add(record, "language", _first_text(unit.metadata, _LANGUAGE_KEYS))
    for value in _metadata_values(unit.metadata, _RELATION_KEYS):
        _add(record, "relation", value)
    return record


def _add(record: ET.Element, name: str, value: Any) -> None:
    text = _clean_text(_xml_value(value))
    if not text:
        return
    element = ET.SubElement(record, f"{{{DC_NS}}}{name}")
    element.text = text


def _metadata_values(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = _nested_value(metadata, key)
        if value is None:
            continue
        values.extend(_list_text(value))
    return values


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    values = _metadata_values(metadata, keys)
    return values[0] if values else ""


def _nested_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata.get(key)
    current: Any = metadata
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current.get(part)
    return current


def _list_text(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [text for item in value if (text := _clean_text(_xml_value(item)))]
    return [text] if (text := _clean_text(_xml_value(value))) else []


def _xml_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _xml_value(value.model_dump())
    if isinstance(value, Mapping):
        for key in ("name", "literal", "title", "value", "id", "doi", "url"):
            if key in value:
                return _xml_value(value.get(key))
        return "; ".join(
            f"{key}: {_xml_value(item)}"
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if _xml_value(item)
        )
    return str(value)


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (str(_xml_value(unit.source_project)), str(unit.source_id or ""), str(unit.title or ""))
