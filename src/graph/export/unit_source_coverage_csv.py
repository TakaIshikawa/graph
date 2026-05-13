"""CSV export for unit source coverage by source project."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "unit_count",
    "unique_source_id_count",
    "missing_source_id_count",
    "url_coverage_percent",
    "tag_coverage_percent",
    "earliest_unit_date",
    "latest_unit_date",
    "duplicate_source_id_count",
]
_URL_KEYS = {"url", "urls", "source_url", "source_urls", "link", "links"}
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://[^\s<>()\]]+")


def export_units_to_source_coverage_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source coverage statistics grouped by source project."""
    unit_list = list(units)
    rows = _coverage_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_project_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _coverage_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[_unit_source(unit)].append(unit)

    rows: list[dict[str, str | int]] = []
    for source_project in sorted(groups, key=_sort_key):
        source_units = groups[source_project]
        source_ids = [_source_id(unit) for unit in source_units]
        present_source_ids = [source_id for source_id in source_ids if source_id]
        source_id_counts = Counter(present_source_ids)
        unit_dates = [unit_date for unit in source_units for unit_date in _unit_dates(unit)]
        unit_count = len(source_units)

        rows.append(
            {
                "source_project": source_project,
                "unit_count": unit_count,
                "unique_source_id_count": len(source_id_counts),
                "missing_source_id_count": len(source_units) - len(present_source_ids),
                "url_coverage_percent": _decimal(_coverage_percent(source_units, _has_url)),
                "tag_coverage_percent": _decimal(_coverage_percent(source_units, _has_tags)),
                "earliest_unit_date": min(unit_dates).isoformat() if unit_dates else "",
                "latest_unit_date": max(unit_dates).isoformat() if unit_dates else "",
                "duplicate_source_id_count": sum(count - 1 for count in source_id_counts.values() if count > 1),
            }
        )
    return rows


def _coverage_percent(units: list[KnowledgeUnit], predicate: Any) -> float:
    if not units:
        return 0.0
    return sum(1 for unit in units if predicate(unit)) * 100 / len(units)


def _has_url(unit: KnowledgeUnit) -> bool:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    for raw_key, value in metadata.items():
        if _key(raw_key) in _URL_KEYS and _contains_url(value):
            return True
    return False


def _contains_url(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_url(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return any(_contains_url(item) for item in value)
    text = _inline_text(value)
    if not text:
        return False
    candidates = _URL_RE.findall(text) or [text]
    return any(_is_url(candidate) for candidate in candidates)


def _is_url(value: object) -> bool:
    parsed = urlparse(_inline_text(value))
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.netloc)


def _has_tags(unit: KnowledgeUnit) -> bool:
    return any(_inline_text(tag) for tag in (unit.tags or []))


def _unit_dates(unit: KnowledgeUnit) -> list[date]:
    dates = []
    for value in (
        getattr(unit, "created_at", None),
        getattr(unit, "ingested_at", None),
        getattr(unit, "updated_at", None),
    ):
        unit_date = _date_value(value)
        if unit_date is not None:
            dates.append(unit_date)
    return dates


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _inline_text(value)
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


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _source_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_id)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _key(value: object) -> str:
    return _field_value(value).casefold().replace("-", "_").replace(" ", "_")


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
