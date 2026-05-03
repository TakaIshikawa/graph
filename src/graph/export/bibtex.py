"""BibTeX export helpers for knowledge units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

_AUTHOR_KEYS = ("authors", "author", "creators", "creator")
_URL_KEYS = ("url", "source_url", "external_url", "uri", "doi")
_DATE_KEYS = (
    "year",
    "publication_year",
    "published_year",
    "date",
    "publication_date",
    "published_at",
    "issued",
)
_TITLE_KEYS = ("title", "publication_title", "book_title")
_PUBLISHER_KEYS = ("publisher", "publisher_name")
_JOURNAL_KEYS = ("journal", "journal_title", "container_title")
_ABSTRACT_KEYS = ("abstract", "summary", "description")


def export_units_to_bibtex(units: Iterable[KnowledgeUnit]) -> str:
    """Return knowledge units as deterministic BibTeX entries."""
    all_units = list(units)
    exported_units = (
        all_units
        if isinstance(units, Sequence)
        else sorted(all_units, key=_unit_sort_key)
    )

    entries: list[str] = []
    for unit in exported_units:
        entry = _unit_bibtex_entry(unit)
        if entry:
            entries.append(entry)
    return "\n\n".join(entries) + ("\n" if entries else "")


def _unit_bibtex_entry(unit: KnowledgeUnit) -> str:
    """Generate BibTeX entry for a unit."""
    # Title
    title = _first_text(unit.metadata, _TITLE_KEYS) or _clean_text(unit.title)
    if not title:
        # Skip units without a title
        return ""

    entry_type = _determine_entry_type(unit.metadata)
    cite_key = _generate_cite_key(unit)

    fields: list[str] = []
    fields.append(_bibtex_field("title", title))

    # Author(s)
    authors = _authors(unit.metadata)
    if authors:
        fields.append(_bibtex_field("author", " and ".join(authors)))

    # Year
    year = _publication_year(unit.metadata)
    if year:
        fields.append(_bibtex_field("year", year))

    # Entry-type specific fields
    if entry_type == "article":
        journal = _first_text(unit.metadata, _JOURNAL_KEYS)
        if journal:
            fields.append(_bibtex_field("journal", journal))

        volume = _metadata_value(unit.metadata, "volume")
        if volume:
            fields.append(_bibtex_field("volume", volume))

        number = _metadata_value(unit.metadata, "number", "issue")
        if number:
            fields.append(_bibtex_field("number", number))

        pages = _metadata_value(unit.metadata, "pages", "page")
        if pages:
            fields.append(_bibtex_field("pages", pages))

    elif entry_type == "book":
        publisher = _first_text(unit.metadata, _PUBLISHER_KEYS)
        if publisher:
            fields.append(_bibtex_field("publisher", publisher))

        edition = _metadata_value(unit.metadata, "edition")
        if edition:
            fields.append(_bibtex_field("edition", edition))

        isbn = _metadata_value(unit.metadata, "isbn")
        if isbn:
            fields.append(_bibtex_field("isbn", isbn))

    # Common fields for all types
    url = _first_text(unit.metadata, _URL_KEYS)
    if url:
        if url.startswith("10."):
            fields.append(_bibtex_field("doi", url))
        else:
            fields.append(_bibtex_field("url", url))

    abstract = _abstract(unit)
    if abstract:
        fields.append(_bibtex_field("abstract", abstract))

    note = _metadata_value(unit.metadata, "note", "notes")
    if note:
        fields.append(_bibtex_field("note", note))

    # Build the entry
    if not fields:
        return ""

    field_lines = ",\n  ".join(fields)
    return f"@{entry_type}{{{cite_key},\n  {field_lines}\n}}"


def _determine_entry_type(metadata: Mapping[str, Any]) -> str:
    """Determine BibTeX entry type from metadata."""
    entry_type = _metadata_value(metadata, "entry_type", "type", "publication_type")
    if entry_type:
        entry_type_lower = entry_type.lower()
        if "article" in entry_type_lower or "journal" in entry_type_lower:
            return "article"
        if "book" in entry_type_lower:
            return "book"
        if "online" in entry_type_lower or "web" in entry_type_lower:
            return "online"

    # Infer from available fields
    if _first_text(metadata, _JOURNAL_KEYS):
        return "article"
    if _first_text(metadata, _PUBLISHER_KEYS):
        return "book"
    if _first_text(metadata, _URL_KEYS):
        return "online"

    return "misc"


def _generate_cite_key(unit: KnowledgeUnit) -> str:
    """Generate a citation key from unit metadata."""
    authors = _authors(unit.metadata)
    year = _publication_year(unit.metadata)
    title = _first_text(unit.metadata, _TITLE_KEYS) or _clean_text(unit.title)

    parts: list[str] = []

    # First author's last name or first word from author
    if authors:
        first_author = authors[0]
        # Extract last name (assume "Last, First" or "First Last" format)
        if "," in first_author:
            last_name = first_author.split(",")[0].strip()
        else:
            # Take the first word if there's no comma
            words = first_author.split()
            last_name = words[0] if words else first_author
        author_key = _sanitize_key_part(last_name)
        if author_key:
            parts.append(author_key)

    # Year
    if year:
        parts.append(year)

    # First significant word from title
    if title:
        words = title.split()
        # Skip common articles
        skip_words = {"a", "an", "the", "of", "in", "on", "at", "to", "for"}
        for word in words:
            word_clean = _sanitize_key_part(word)
            if word_clean and word_clean.lower() not in skip_words:
                parts.append(word_clean[:15])  # Limit length
                break

    # Ensure we have at least something
    if not parts:
        parts.append(_sanitize_key_part(unit.id))

    return "".join(parts)


def _sanitize_key_part(text: str) -> str:
    """Sanitize text for use in BibTeX citation key."""
    # Remove special characters, keep only alphanumeric
    text = re.sub(r"[^a-zA-Z0-9]", "", text)
    return text


def _bibtex_field(key: str, value: str) -> str:
    """Format a BibTeX field."""
    # Escape special characters
    escaped = _escape_bibtex(value)
    return f"{key} = {{{escaped}}}"


def _escape_bibtex(text: str) -> str:
    """Escape special BibTeX characters."""
    # Replace special characters with their LaTeX equivalents
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


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
    # Don't use content as fallback for abstract in BibTeX
    return ""


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key not in metadata:
            continue
        value = _clean_text(_scalar_text(metadata.get(key)))
        if value:
            return value
    return ""


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> str:
    """Get first non-empty metadata value from given keys."""
    for key in keys:
        if key in metadata:
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


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        str(unit.source_project or ""),
        str(unit.source_id or ""),
        str(unit.title or ""),
    )
