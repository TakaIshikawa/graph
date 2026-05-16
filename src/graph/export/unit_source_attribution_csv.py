"""CSV export for units with incomplete source attribution."""

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
    "source_project",
    "attribution_score",
    "missing_fields",
    "present_fields",
    "source_url",
    "author",
    "citation",
]
_REQUIRED_FIELDS = ("source_project", "source_id", "source_url", "author", "provenance")
_SOURCE_URL_KEYS = ("source_url", "url")
_AUTHOR_KEYS = ("author", "creator")
_CITATION_KEYS = ("citation",)
_PROVENANCE_KEYS = ("citation", "file_path", "imported_from")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_source_attribution_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write units that have incomplete source attribution."""
    unit_list = list(units)
    rows = _attribution_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "weak_unit_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _attribution_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        present = _present_fields(unit)
        if len(present) == len(_REQUIRED_FIELDS):
            continue

        missing = [field for field in _REQUIRED_FIELDS if field not in present]
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "source_project": _field_value(_get(unit, "source_project")) or "Unknown",
                "attribution_score": f"{len(present) / len(_REQUIRED_FIELDS):.2f}",
                "missing_fields": "; ".join(missing),
                "present_fields": "; ".join(field for field in _REQUIRED_FIELDS if field in present),
                "source_url": _first_value(unit, _SOURCE_URL_KEYS),
                "author": _first_value(unit, _AUTHOR_KEYS),
                "citation": _first_value(unit, _CITATION_KEYS),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            float(row["attribution_score"]),
            _sort_key(row["source_project"]),
            _sort_key(row["unit_id"]),
            _sort_key(row["title"]),
        ),
    )


def _present_fields(unit: KnowledgeUnit | Mapping[str, Any]) -> set[str]:
    present: set[str] = set()
    if _field_value(_get(unit, "source_project")):
        present.add("source_project")
    if _field_value(_get(unit, "source_id")):
        present.add("source_id")
    if _first_value(unit, _SOURCE_URL_KEYS):
        present.add("source_url")
    if _first_value(unit, _AUTHOR_KEYS):
        present.add("author")
    if _first_value(unit, _PROVENANCE_KEYS):
        present.add("provenance")
    return present


def _first_value(unit: KnowledgeUnit | Mapping[str, Any], keys: tuple[str, ...]) -> str:
    metadata = _metadata(unit)
    for key in keys:
        text = _metadata_text(_get(unit, key))
        if text:
            return text
        text = _metadata_text(_casefold_get(metadata, key))
        if text:
            return text
    return ""


def _metadata_text(value: object) -> str:
    if value is None or isinstance(value, bytes):
        return ""
    if isinstance(value, str):
        return _field_value(value)
    if isinstance(value, Mapping):
        return ""
    if isinstance(value, Iterable):
        values = [_metadata_text(item) for item in value]
        return "; ".join(value for value in values if value)
    return _field_value(value)


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _render_csv(rows: list[dict[str, str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
