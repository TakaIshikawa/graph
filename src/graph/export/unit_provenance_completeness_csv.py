"""CSV export for unit provenance completeness."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "source_project", "source_entity_type", "score", "missing_fields", "provenance_url"]
_REQUIRED_FIELDS = ("source_project", "source_id", "source_entity_type", "provenance_url", "author_or_account", "source_date")
_URL_KEYS = ("url", "permalink", "source_url", "external_url", "canonical_url", "web_url")
_AUTHOR_KEYS = ("author", "creator", "account", "username", "user", "owner")
_DATE_KEYS = ("created_at", "updated_at", "imported_at", "published_at", "date", "source_date")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_provenance_completeness_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write traceability scores for units."""
    unit_list = list(units)
    rows = _completeness_rows(unit_list)
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


def _completeness_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        present = _present_fields(unit)
        missing = [field for field in _REQUIRED_FIELDS if field not in present]
        rows.append(
            {
                "unit_id": _unit_id(unit),
                "source_project": _field_value(_get(unit, "source_project")) or "Unknown",
                "source_entity_type": _field_value(_get(unit, "source_entity_type")) or "Unknown",
                "score": f"{len(present) / len(_REQUIRED_FIELDS):.2f}",
                "missing_fields": "; ".join(missing),
                "provenance_url": _first_metadata_value(unit, _URL_KEYS),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            float(row["score"]),
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
            _sort_key(row["unit_id"]),
        ),
    )


def _present_fields(unit: KnowledgeUnit | Mapping[str, Any]) -> set[str]:
    present: set[str] = set()
    if _field_value(_get(unit, "source_project")):
        present.add("source_project")
    if _field_value(_get(unit, "source_id")):
        present.add("source_id")
    if _field_value(_get(unit, "source_entity_type")):
        present.add("source_entity_type")
    if _first_metadata_value(unit, _URL_KEYS):
        present.add("provenance_url")
    if _first_metadata_value(unit, _AUTHOR_KEYS):
        present.add("author_or_account")
    if _first_date_value(unit):
        present.add("source_date")
    return present


def _first_date_value(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    for key in _DATE_KEYS:
        text = _value_text(_get(unit, key))
        if text:
            return text
    return _first_metadata_value(unit, _DATE_KEYS)


def _first_metadata_value(unit: KnowledgeUnit | Mapping[str, Any], keys: tuple[str, ...]) -> str:
    metadata = _metadata(unit)
    for key in keys:
        text = _value_text(_get(unit, key))
        if text:
            return text
        value = _casefold_get(metadata, key)
        text = _value_text(value)
        if text:
            return text
    return ""


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _value_text(value: object) -> str:
    if value is None or isinstance(value, bytes):
        return ""
    if isinstance(value, str):
        return _field_value(value)
    if isinstance(value, Mapping):
        return ""
    if isinstance(value, Iterable):
        values = [_value_text(item) for item in value]
        return "; ".join(value for value in values if value)
    return _field_value(value)


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


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
