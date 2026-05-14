"""CSV export for source date precision coverage."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "unit_count",
    "dated_unit_count",
    "undated_unit_count",
    "date_only_count",
    "datetime_count",
    "year_month_count",
    "year_only_count",
    "date_coverage_percent",
]
_DATE_FIELDS = ("created_at", "updated_at")
_DATE_KEY_RE = re.compile(r"(date|time|year|_at$)", re.IGNORECASE)
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_YEAR_ONLY_RE = re.compile(r"^\d{4}$")
_PRECISION_ORDER = {
    "undated": 0,
    "year_only": 1,
    "year_month": 2,
    "date_only": 3,
    "datetime": 4,
}
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_date_precision_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write source date precision statistics as deterministic CSV."""
    unit_list = list(units)
    rows = _precision_rows(unit_list)
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


def _precision_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        groups[_unit_source(unit)][_unit_precision(unit)] += 1

    rows: list[dict[str, str | int]] = []
    for source_project in sorted(groups, key=_sort_key):
        counts = groups[source_project]
        unit_count = sum(counts.values())
        undated_count = counts["undated"]
        dated_count = unit_count - undated_count
        rows.append(
            {
                "source_project": source_project,
                "unit_count": unit_count,
                "dated_unit_count": dated_count,
                "undated_unit_count": undated_count,
                "date_only_count": counts["date_only"],
                "datetime_count": counts["datetime"],
                "year_month_count": counts["year_month"],
                "year_only_count": counts["year_only"],
                "date_coverage_percent": _decimal(dated_count * 100 / unit_count),
            }
        )
    return rows


def _unit_precision(unit: KnowledgeUnit) -> str:
    precisions: list[str] = []
    precisions.extend(_date_precision(getattr(unit, field, None)) for field in _DATE_FIELDS)

    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    for key, value in metadata.items():
        if _DATE_KEY_RE.search(_inline_text(key)):
            precisions.extend(_date_precision(item) for item in _iter_values(value))

    valid = [precision for precision in precisions if precision is not None]
    if not valid:
        return "undated"
    return max(valid, key=lambda precision: _PRECISION_ORDER[precision])


def _iter_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return list(value)
    if value is None:
        return []
    return [value]


def _date_precision(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date_only"

    text = _inline_text(value)
    if not text:
        return None
    if _YEAR_ONLY_RE.fullmatch(text):
        return "year_only"
    if _YEAR_MONTH_RE.fullmatch(text):
        return "year_month"
    if _DATE_ONLY_RE.fullmatch(text):
        return "date_only"
    if _DATETIME_RE.fullmatch(text):
        return "datetime"
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
