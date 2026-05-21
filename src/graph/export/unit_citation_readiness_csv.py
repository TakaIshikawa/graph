"""CSV export for unit citation readiness."""

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
    "source_project",
    "has_title",
    "has_creator",
    "has_date",
    "has_url",
    "has_doi",
    "has_isbn",
    "missing_fields",
    "readiness_score",
]
_CREATOR_KEYS = ("author", "authors", "creator", "creators", "created_by")
_DATE_KEYS = ("date", "year", "published", "published_at", "publication_date", "created_at", "updated_at")
_URL_KEYS = ("url", "source_url", "external_url", "canonical_url", "link")
_DOI_KEYS = ("doi",)
_ISBN_KEYS = ("isbn", "isbn10", "isbn13")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_citation_readiness_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write citation-ready metadata flags for each unit."""
    unit_list = list(units)
    rows = _readiness_rows(unit_list)
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


def _readiness_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        present = _present_fields(unit)
        missing = [field for field in ("title", "creator", "date", "url_or_identifier") if field not in present]
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "source_project": _field_value(_get(unit, "source_project")) or "Unknown",
                "has_title": _flag("title" in present),
                "has_creator": _flag("creator" in present),
                "has_date": _flag("date" in present),
                "has_url": _flag("url" in present),
                "has_doi": _flag("doi" in present),
                "has_isbn": _flag("isbn" in present),
                "missing_fields": ";".join(missing),
                "readiness_score": f"{(4 - len(missing)) / 4:.2f}",
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["unit_id"])))


def _present_fields(unit: KnowledgeUnit | Mapping[str, Any]) -> set[str]:
    metadata = _metadata(unit)
    mappings = [metadata, *_source_mappings(metadata)]
    present: set[str] = set()
    if _populated(_get(unit, "title")) or any(_has_value(mapping, ("title", "name")) for mapping in mappings):
        present.add("title")
    if any(_has_value(mapping, _CREATOR_KEYS) for mapping in mappings):
        present.add("creator")
    if any(_has_value(mapping, _DATE_KEYS) for mapping in mappings) or _populated(_get(unit, "created_at")):
        present.add("date")
    if any(_has_value(mapping, _URL_KEYS) for mapping in mappings):
        present.add("url")
    if any(_has_value(mapping, _DOI_KEYS) for mapping in mappings):
        present.add("doi")
    if any(_has_value(mapping, _ISBN_KEYS) for mapping in mappings):
        present.add("isbn")
    if "url" in present or "doi" in present or "isbn" in present:
        present.add("url_or_identifier")
    return present


def _source_mappings(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[object] = []
    for key in ("source", "sources", "citation", "citations"):
        values.extend(_flat_values(_casefold_get(metadata, key)))
    return [value for value in values if isinstance(value, Mapping)]


def _flat_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _flat_values(entry)]
    return [value]


def _has_value(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return any(_populated(_casefold_get(mapping, key)) for key in keys)


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(_populated(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(_populated(item) for item in value)
    return bool(_field_value(value))


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _flag(value: bool) -> str:
    return "true" if value else "false"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)

