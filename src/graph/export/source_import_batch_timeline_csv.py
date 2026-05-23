"""CSV export for source import batch timelines."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

_FIELDNAMES = [
    "import_batch",
    "imported_date",
    "source_count",
    "earliest_source_date",
    "latest_source_date",
    "source_names",
]
_UNKNOWN_BATCH = "unknown_batch"
_UNKNOWN_DATE = "unknown_date"
_WHITESPACE_RE = re.compile(r"\s+")
_IMPORTED_AT_KEYS = ("imported_at", "imported_date", "ingested_at", "created_at")
_SOURCE_DATE_KEYS = ("source_date", "published_at", "published_date", "date", "created_at", "updated_at")


def export_source_import_batch_timeline_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source import batches grouped by batch and imported date."""
    source_list = list(sources)
    rows = _timeline_rows(source_list)
    text = _render_csv(rows)
    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "source_count": len(source_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _timeline_rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any] | object]] = defaultdict(list)
    for source in sources:
        batch = _lookup(source, "import_batch") or _UNKNOWN_BATCH
        imported_date = _date_label(_first_date(source, _IMPORTED_AT_KEYS))
        groups[(batch, imported_date)].append(source)

    rows: list[dict[str, str | int]] = []
    for (batch, imported_date), grouped_sources in sorted(
        groups.items(),
        key=lambda item: (_date_sort_key(item[0][1]), _sort_key(item[0][0])),
    ):
        source_dates = sorted(
            source_date for source in grouped_sources if (source_date := _first_date(source, _SOURCE_DATE_KEYS)) is not None
        )
        rows.append(
            {
                "import_batch": batch,
                "imported_date": imported_date,
                "source_count": len(grouped_sources),
                "earliest_source_date": source_dates[0].isoformat() if source_dates else _UNKNOWN_DATE,
                "latest_source_date": source_dates[-1].isoformat() if source_dates else _UNKNOWN_DATE,
                "source_names": "; ".join(_source_names(grouped_sources)),
            }
        )
    return rows


def _source_names(sources: list[Mapping[str, Any] | object]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for source in sorted(sources, key=lambda value: (_sort_key(_source_name(value)), _sort_key(_source_id(value)))):
        name = _source_name(source) or _source_id(source) or "Unnamed source"
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _lookup(source: Mapping[str, Any] | object, key: str) -> str:
    direct = _field_value(_get(source, key))
    if direct:
        return direct
    metadata = _get(source, "metadata")
    if isinstance(metadata, Mapping):
        return _field_value(_casefold_get(metadata, key))
    return ""


def _first_date(source: Mapping[str, Any] | object, keys: tuple[str, ...]) -> date | None:
    for key in keys:
        parsed = _date_value(_lookup(source, key))
        if parsed is not None:
            return parsed
    return None


def _date_value(value: object) -> date | None:
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


def _date_label(value: date | None) -> str:
    return value.isoformat() if value else _UNKNOWN_DATE


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _source_id(source: Mapping[str, Any] | object) -> str:
    return _field_value(_get(source, "id")) or _field_value(_get(source, "source_id"))


def _source_name(source: Mapping[str, Any] | object) -> str:
    return _field_value(_get(source, "name")) or _field_value(_get(source, "title"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _date_sort_key(value: object) -> tuple[int, str]:
    text = _inline_text(value)
    return (1, text) if text == _UNKNOWN_DATE else (0, text)
