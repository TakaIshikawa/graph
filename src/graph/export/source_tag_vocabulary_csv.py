"""CSV export for source tag vocabulary summaries."""

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
    "normalized_tag",
    "raw_tag_variants",
    "unit_count",
    "first_seen",
    "last_seen",
    "representative_unit_ids",
]
_UNKNOWN = "Unknown"
_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_METADATA_DATE_KEYS = (
    "observed_at",
    "observed_date",
    "source_date",
    "published_at",
    "published_date",
    "created_at",
    "updated_at",
    "date",
)
_METADATA_SOURCE_KEYS = ("source_project", "project", "source")
_METADATA_TAG_KEYS = ("tags", "tag")
_SEPARATOR_RE = re.compile(r"[\s\-_:/|.,;]+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_tag_vocabulary_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | Any | None = None,
) -> str | dict[str, Any]:
    """Return or write tag vocabulary rows grouped by source project."""
    unit_list = list(units)
    rows = _vocabulary_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    stats = {
        "unit_count": len(unit_list),
        "tag_rows_exported": len(rows),
        "source_count": len({row["source_project"] for row in rows}),
    }
    return _write_output(path, text, stats)


def _vocabulary_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"raw_tags": set(), "unit_ids": set(), "dates": []}
    )

    for unit in units:
        source = _unit_source(unit)
        unit_id = _field_value(_get(unit, "id"))
        unit_date = _unit_date(unit)
        for tag in _unit_tags(unit):
            normalized = _normalized_tag(tag)
            if not normalized:
                continue
            group = groups[(source, normalized)]
            group["raw_tags"].add(tag)
            if unit_id:
                group["unit_ids"].add(unit_id)
            if unit_date is not None:
                group["dates"].append(unit_date)

    rows: list[dict[str, str | int]] = []
    for (source, normalized), group in groups.items():
        dates = sorted(group["dates"])
        rows.append(
            {
                "source_project": source,
                "normalized_tag": normalized,
                "raw_tag_variants": _joined(group["raw_tags"]),
                "unit_count": len(group["unit_ids"]),
                "first_seen": dates[0].isoformat() if dates else "",
                "last_seen": dates[-1].isoformat() if dates else "",
                "representative_unit_ids": _joined(group["unit_ids"]),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source_project"]),
            _sort_key(row["normalized_tag"]),
            _sort_key(row["representative_unit_ids"]),
        ),
    )


def _unit_source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    value = _field_value(_get(unit, "source_project"))
    metadata = _metadata(unit)
    for key in _METADATA_SOURCE_KEYS:
        value = value or _field_value(metadata.get(key))
    return value or _UNKNOWN


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values: list[object] = []
    unit_tags = _get(unit, "tags")
    if isinstance(unit_tags, str):
        values.append(unit_tags)
    elif isinstance(unit_tags, Iterable):
        values.extend(unit_tags)

    metadata = _metadata(unit)
    for key in _METADATA_TAG_KEYS:
        metadata_tags = metadata.get(key)
        if isinstance(metadata_tags, str):
            values.append(metadata_tags)
        elif isinstance(metadata_tags, Iterable):
            values.extend(metadata_tags)

    return sorted({_field_value(tag) for tag in values if _field_value(tag)}, key=_sort_key)


def _unit_date(unit: KnowledgeUnit | Mapping[str, Any]) -> date | None:
    metadata = _metadata(unit)
    for key in _METADATA_DATE_KEYS:
        parsed = _date_value(metadata.get(key))
        if parsed is not None:
            return parsed
    for field in _DATE_FIELDS:
        parsed = _date_value(_get(unit, field))
        if parsed is not None:
            return parsed
    return None


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _date_value(value: object) -> date | None:
    if value is None:
        return None
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


def _normalized_tag(value: object) -> str:
    return _SEPARATOR_RE.sub(" ", _field_value(value).casefold()).strip()


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
