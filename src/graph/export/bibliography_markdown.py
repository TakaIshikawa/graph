"""Markdown bibliography export helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, overload

from pydantic import BaseModel

from graph.export.csl_json import _authors as csl_authors
from graph.export.csl_json import _issued
from graph.types.models import KnowledgeUnit

URL_KEYS = ("URL", "url", "source_url", "external_url", "uri")
DOI_KEYS = ("DOI", "doi", "digital_object_identifier")
ISBN_KEYS = ("ISBN", "isbn")
CONTAINER_KEYS = ("container-title", "container_title", "journal", "booktitle")


@overload
def export_units_to_bibliography_markdown(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: None = None,
) -> str: ...


@overload
def export_units_to_bibliography_markdown(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path,
) -> dict[str, Any]: ...


def export_units_to_bibliography_markdown(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write deterministic Markdown bibliography bullets."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    exported_units = sorted(unit_list, key=_unit_key)

    lines = ["# Bibliography", ""]
    for unit in exported_units:
        lines.append(f"- {_citation(unit)}")
        abstract = _first_text(unit.metadata, ("abstract", "summary", "description"))
        abstract = abstract or _clean_text(unit.content)
        if abstract:
            lines.append(f"  - {abstract}")
    text = "\n".join(lines).rstrip() + "\n"

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(exported_units),
        "bytes_written": output_path.stat().st_size,
    }


def _citation(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata
    parts: list[str] = []
    authors = _author_texts(metadata)
    if authors:
        parts.append(_join_authors(authors))
    year = _year(metadata)
    title = _first_text(metadata, ("title",)) or _clean_text(unit.title) or "Untitled"
    parts.append(f"({year}). {title}." if year else f"{title}.")

    container = _first_text(metadata, CONTAINER_KEYS)
    if container:
        parts.append(f"*{container}*.")
    publisher = _first_text(metadata, ("publisher",))
    if publisher:
        parts.append(f"{publisher}.")
    doi = _first_text(metadata, DOI_KEYS)
    if doi:
        parts.append(f"DOI: {doi}.")
    isbn = _first_text(metadata, ISBN_KEYS)
    if isbn:
        parts.append(f"ISBN: {isbn}.")
    url = _first_text(metadata, URL_KEYS)
    if url:
        parts.append(url)
    return " ".join(parts)


def _author_texts(metadata: Mapping[str, Any]) -> list[str]:
    authors = []
    for author in csl_authors(metadata):
        if literal := author.get("literal"):
            authors.append(literal)
            continue
        name = ", ".join(part for part in (author.get("family"), author.get("given")) if part)
        if name:
            authors.append(name)
    return authors


def _join_authors(authors: list[str]) -> str:
    if len(authors) <= 2:
        return " and ".join(authors)
    return ", ".join(authors[:-1]) + f", and {authors[-1]}"


def _year(metadata: Mapping[str, Any]) -> str:
    issued = _issued(metadata)
    return str(issued[0]) if issued else ""


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _nested_value(metadata, key)
        text = _clean_text(value)
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


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _clean_text(value.model_dump())
    if isinstance(value, Mapping):
        return "; ".join(
            f"{key}: {_clean_text(item)}"
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if _clean_text(item)
        )
    if isinstance(value, list | tuple | set):
        return "; ".join(_clean_text(item) for item in value if _clean_text(item))
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def _unit_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (str(unit.source_project or ""), str(unit.source_id or ""), str(unit.title or ""))
