"""CSL-JSON export helpers for bibliographic knowledge units."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

_AUTHOR_KEYS = ("authors", "author", "creators", "creator")
_DATE_KEYS = ("issued", "published", "published_at", "publication_date", "date", "year")


def export_units_to_csl_json(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str:
    """Return units as a CSL-JSON list."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    records = [_unit_record(unit) for unit in unit_list]
    text = json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2)
    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    return text


def _unit_record(unit: KnowledgeUnit) -> dict[str, Any]:
    metadata = unit.metadata
    record: dict[str, Any] = {
        "id": _first_text(metadata, ("id", "citation_id")) or unit.id or unit.source_id,
        "type": _csl_type(_first_text(metadata, ("type", "entry_type", "item_type"))),
        "title": _first_text(metadata, ("title",)) or unit.title,
    }

    authors = _authors(metadata)
    if authors:
        record["author"] = authors

    issued = _issued(metadata)
    if issued:
        record["issued"] = {"date-parts": [issued]}

    for csl_key, keys in {
        "DOI": ("DOI", "doi", "digital_object_identifier"),
        "ISBN": ("ISBN", "isbn"),
        "URL": ("URL", "url", "source_url", "external_url", "uri"),
        "container-title": ("container-title", "container_title", "journal", "booktitle"),
        "publisher": ("publisher",),
    }.items():
        value = _first_text(metadata, keys)
        if value:
            record[csl_key] = value

    abstract = _first_text(metadata, ("abstract", "summary", "description")) or _clean_text(unit.content)
    if abstract:
        record["abstract"] = abstract
    return record


def _authors(metadata: Mapping[str, Any]) -> list[dict[str, str]]:
    for key in _AUTHOR_KEYS:
        if key in metadata:
            value = metadata.get(key)
            items = value if isinstance(value, list | tuple | set) else [value]
            return [author for item in items if (author := _author(item))]
    return []


def _author(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        family = _clean_text(_scalar_text(value.get("family") or value.get("last")))
        given = _clean_text(_scalar_text(value.get("given") or value.get("first")))
        literal = _clean_text(_scalar_text(value.get("literal") or value.get("name")))
        if family or given:
            author: dict[str, str] = {}
            if family:
                author["family"] = family
            if given:
                author["given"] = given
            return author
        if literal:
            return {"literal": literal}
        return {}
    text = _clean_text(_scalar_text(value))
    if not text:
        return {}
    if "," in text:
        family, given = [part.strip() for part in text.split(",", 1)]
        return {key: part for key, part in {"family": family, "given": given}.items() if part}
    return {"literal": text}


def _issued(metadata: Mapping[str, Any]) -> list[int]:
    for key in _DATE_KEYS:
        if key in metadata:
            parts = _date_parts(metadata.get(key))
            if parts:
                return parts
    return []


def _date_parts(value: Any) -> list[int]:
    if isinstance(value, datetime | date):
        return [value.year, value.month, value.day]
    text = _clean_text(_scalar_text(value))
    if not text:
        return []
    match = re.match(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", text)
    if not match:
        return []
    return [int(part) for part in match.groups() if part]


def _csl_type(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    return {
        "article": "article-journal",
        "journal": "article-journal",
        "inproceedings": "paper-conference",
        "conference": "paper-conference",
        "web": "webpage",
        "online": "webpage",
    }.get(normalized, normalized or "article")


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _nested_value(metadata, key)
        text = _clean_text(_scalar_text(value))
        if text:
            return text
    return ""


def _nested_value(metadata: Mapping[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata.get(key)
    current: Any = metadata
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current.get(part)
    return current


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
