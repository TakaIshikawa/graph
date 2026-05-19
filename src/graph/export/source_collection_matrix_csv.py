"""CSV export for source collection distribution."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "collection",
    "source_entity_type",
    "unit_count",
    "tagged_unit_count",
    "first_seen",
    "last_seen",
    "representative_unit_ids",
]
_UNKNOWN = "Unknown"
_COLLECTION_KEYS = ("collection", "collections", "list", "list_name", "project", "folder", "notebook", "shelf", "playlist")
_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_DATE_KEYS = ("date", "source_date", "observed_at", "created_at", "updated_at", "published_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_collection_matrix_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write collection rows grouped by source project and entity type."""
    unit_list = list(units)
    rows = _matrix_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    return _write_output(path, text, {"unit_count": len(unit_list), "rows_exported": len(rows)})


def _matrix_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"unit_ids": set(), "tagged_ids": set(), "dates": []}
    )
    for unit in units:
        collections = _unit_collections(unit)
        if not collections:
            continue
        unit_id = _unit_id(unit)
        unit_date = _unit_date(unit)
        has_tags = bool(_unit_tags(unit))
        for collection in collections:
            group = groups[(_unit_source(unit), collection, _unit_source_type(unit))]
            if unit_id:
                group["unit_ids"].add(unit_id)
                if has_tags:
                    group["tagged_ids"].add(unit_id)
            if unit_date is not None:
                group["dates"].append(unit_date)

    rows: list[dict[str, str | int]] = []
    for (source, collection, entity_type), group in groups.items():
        dates = sorted(group["dates"])
        rows.append(
            {
                "source_project": source,
                "collection": collection,
                "source_entity_type": entity_type,
                "unit_count": len(group["unit_ids"]),
                "tagged_unit_count": len(group["tagged_ids"]),
                "first_seen": dates[0].isoformat() if dates else "",
                "last_seen": dates[-1].isoformat() if dates else "",
                "representative_unit_ids": _joined(group["unit_ids"]),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["collection"]), _sort_key(row["source_entity_type"])))


def _unit_collections(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values: list[object] = []
    if hasattr(unit, "collection"):
        values.append(getattr(unit, "collection"))
    metadata = _metadata(unit)
    for key in _COLLECTION_KEYS:
        values.extend(_iter_values(metadata.get(key)))
    return sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key)


def _iter_values(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    tags = _get(unit, "tags", [])
    if isinstance(tags, str):
        return [_field_value(tags)] if _field_value(tags) else []
    if isinstance(tags, Iterable):
        return [_field_value(tag) for tag in tags if _field_value(tag)]
    return []


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    for key in _DATE_KEYS:
        if (parsed := _date_value(metadata.get(key))) is not None:
            return parsed
    for field in _DATE_FIELDS:
        if (parsed := _date_value(_get(unit, field))) is not None:
            return parsed
    return None


def _date_value(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _field_value(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _unit_source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_project")) or _UNKNOWN


def _unit_source_type(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_entity_type")) or _UNKNOWN


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_output(path: str | Path | Any, text: str, stats: dict[str, Any]) -> dict[str, Any]:
    if hasattr(path, "write") and not isinstance(path, str | Path):
        written = path.write(text)
        stats["bytes_written"] = len(text.encode("utf-8")) if written is None else written
        return stats
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    stats["path"] = str(output_path)
    stats["bytes_written"] = output_path.stat().st_size
    return stats


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
