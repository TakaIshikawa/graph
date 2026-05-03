"""RIS export helpers for knowledge units."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

_AUTHOR_KEYS = ("authors", "author", "creators", "creator")
_URL_KEYS = ("url", "source_url", "external_url", "uri")
_DATE_KEYS = (
    "year",
    "publication_year",
    "published_year",
    "date",
    "publication_date",
    "published_at",
    "issued",
)
_ABSTRACT_KEYS = ("abstract", "summary", "description")


def export_units_to_ris(units: Iterable[KnowledgeUnit]) -> str:
    """Return knowledge units as deterministic line-oriented RIS records."""
    all_units = list(units)
    exported_units = (
        all_units
        if isinstance(units, Sequence)
        else sorted(all_units, key=_unit_sort_key)
    )

    lines: list[str] = []
    for unit in exported_units:
        lines.extend(_unit_record_lines(unit))
    return "\n".join(lines) + ("\n" if lines else "")


def _unit_record_lines(unit: KnowledgeUnit) -> list[str]:
    lines = [_ris_line("TY", "ELEC")]

    title = _clean_text(unit.title)
    if title:
        lines.append(_ris_line("TI", title))

    for author in _authors(unit.metadata):
        lines.append(_ris_line("AU", author))

    published_year = _publication_year(unit.metadata)
    if published_year:
        lines.append(_ris_line("PY", published_year))

    url = _first_text(unit.metadata, _URL_KEYS)
    if url:
        lines.append(_ris_line("UR", url))

    for tag in _keywords(unit.tags):
        lines.append(_ris_line("KW", tag))

    abstract = _abstract(unit)
    if abstract:
        lines.append(_ris_line("AB", abstract))

    lines.append(_ris_line("ER", ""))
    return lines


def _authors(metadata: Mapping[str, Any]) -> list[str]:
    for key in _AUTHOR_KEYS:
        if key not in metadata:
            continue
        return _list_text(metadata.get(key))
    return []


def _publication_year(metadata: Mapping[str, Any]) -> str:
    for key in _DATE_KEYS:
        if key not in metadata:
            continue
        year = _year_text(metadata.get(key))
        if year:
            return year
    return ""


def _abstract(unit: KnowledgeUnit) -> str:
    abstract = _first_text(unit.metadata, _ABSTRACT_KEYS)
    if abstract:
        return abstract
    return _clean_text(unit.content)


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key not in metadata:
            continue
        value = _clean_text(_scalar_text(metadata.get(key)))
        if value:
            return value
    return ""


def _list_text(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        items = value
    else:
        items = [value]
    return [
        text
        for text in (_clean_text(_author_text(item)) for item in items)
        if text
    ]


def _author_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("name", "full_name", "display_name", "literal", "family"):
            text = _scalar_text(value.get(key))
            if text:
                return text
        return _scalar_text(value)
    return _scalar_text(value)


def _keywords(tags: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for tag in sorted(_clean_text(_scalar_text(item)) for item in tags):
        if tag and tag not in seen:
            seen.add(tag)
            keywords.append(tag)
    return keywords


def _year_text(value: Any) -> str:
    if isinstance(value, datetime | date):
        return f"{value.year:04d}"
    text = _clean_text(_scalar_text(value))
    if not text:
        return ""
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return text


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _scalar_text(value.model_dump())
    if isinstance(value, Mapping):
        return "; ".join(
            f"{key}: {_scalar_text(item)}"
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if _scalar_text(item)
        )
    return str(value)


def _clean_text(value: str) -> str:
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def _ris_line(tag: str, value: str) -> str:
    return f"{tag}  - {value}"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        str(unit.source_project or ""),
        str(unit.source_id or ""),
        str(unit.title or ""),
    )
