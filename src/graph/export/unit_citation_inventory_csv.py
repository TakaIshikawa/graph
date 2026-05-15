"""CSV export for citation-like metadata attached to units."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "citation_field_count",
    "has_doi",
    "has_isbn",
    "has_pmid",
    "has_arxiv_id",
    "has_url",
    "has_author",
    "has_publisher",
    "has_published_date",
]
_FIELD_KEYS = {
    "doi": ("doi",),
    "isbn": ("isbn", "isbn10", "isbn13"),
    "pmid": ("pmid", "pubmed_id"),
    "arxiv_id": ("arxiv_id", "arxiv", "eprint"),
    "url": ("url", "source_url", "external_url", "canonical_url", "link"),
    "author": ("author", "authors", "creator", "creators"),
    "publisher": ("publisher", "publication", "journal", "venue"),
    "published_date": ("published", "published_at", "publication_date", "date", "year"),
}
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_citation_inventory_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write citation metadata presence flags for each unit."""
    unit_list = list(units)
    rows = _inventory_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _inventory_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in sorted(units, key=_unit_sort_key):
        present = _present_fields(unit)
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _inline_text(unit.title),
                "citation_field_count": len(present),
                "has_doi": _flag("doi" in present),
                "has_isbn": _flag("isbn" in present),
                "has_pmid": _flag("pmid" in present),
                "has_arxiv_id": _flag("arxiv_id" in present),
                "has_url": _flag("url" in present),
                "has_author": _flag("author" in present),
                "has_publisher": _flag("publisher" in present),
                "has_published_date": _flag("published_date" in present),
            }
        )
    return rows


def _present_fields(unit: KnowledgeUnit) -> set[str]:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    mappings = [metadata, *_source_mappings(metadata)]
    present: set[str] = set()
    for field, keys in _FIELD_KEYS.items():
        if any(_has_value(mapping, keys) for mapping in mappings):
            present.add(field)
    return present


def _source_mappings(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[object] = []
    if "source" in metadata:
        values.extend(_flat_values(metadata.get("source")))
    if "sources" in metadata:
        values.extend(_flat_values(metadata.get("sources")))
    return [value for value in values if isinstance(value, Mapping)]


def _flat_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _flat_values(entry)]
    return [value]


def _has_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if _populated(metadata.get(key)):
            return True
    return False


def _populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_inline_text(value))
    if isinstance(value, Mapping):
        return any(_populated(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(_populated(item) for item in value)
    return bool(_inline_text(value))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id) or _inline_text(unit.source_id)


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_id(unit)), _sort_key(unit.title))
