"""CSV export for date lag across graph relations."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "relation",
    "source_unit_id",
    "target_unit_id",
    "source_date",
    "target_date",
    "lag_days",
    "lag_bucket",
]
_DATE_FIELDS = ("created_at", "ingested_at", "updated_at")
_METADATA_DATE_KEYS = ("observed_at", "observed_date", "source_date", "date", "published_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_date_lag_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write signed date lags between edge source and target units."""
    unit_list = list(units)
    edge_list = list(edges)
    rows = _lag_rows(unit_list, edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "edge_count": len(edge_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _lag_rows(units: list[KnowledgeUnit], edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    unit_dates = {_field_value(unit.id): _best_date(unit) for unit in units}
    rows: list[dict[str, str | int]] = []
    for edge in edges:
        source_id = _field_value(edge.from_unit_id)
        target_id = _field_value(edge.to_unit_id)
        source_date = unit_dates.get(source_id)
        target_date = unit_dates.get(target_id)
        lag_days = (target_date - source_date).days if source_date and target_date else None
        rows.append(
            {
                "relation": _field_value(edge.relation),
                "source_unit_id": source_id,
                "target_unit_id": target_id,
                "source_date": source_date.isoformat() if source_date else "",
                "target_date": target_date.isoformat() if target_date else "",
                "lag_days": lag_days if lag_days is not None else "",
                "lag_bucket": _lag_bucket(lag_days),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["relation"]),
            _sort_key(row["source_unit_id"]),
            _sort_key(row["target_unit_id"]),
        ),
    )


def _best_date(unit: KnowledgeUnit) -> date | None:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    values = [_date_value(metadata.get(key)) for key in _METADATA_DATE_KEYS]
    values.extend(_date_value(getattr(unit, field, None)) for field in _DATE_FIELDS)
    dates = [value for value in values if value is not None]
    return min(dates) if dates else None


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


def _lag_bucket(lag_days: int | None) -> str:
    if lag_days is None:
        return "unknown"
    if lag_days == 0:
        return "same_day"
    if lag_days > 0:
        return "source_before_target"
    return "source_after_target"


def _render_csv(rows: list[dict[str, str | int]]) -> str:
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
