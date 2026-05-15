"""CSV export for temporal lag between edge endpoints."""

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
    "edge_id",
    "relation",
    "from_unit_id",
    "to_unit_id",
    "from_date",
    "to_date",
    "lag_days",
    "lag_direction",
]
_UNIT_DATE_FIELDS = ("created_at", "updated_at", "ingested_at")
_METADATA_DATE_KEYS = (
    "date",
    "published_at",
    "published_date",
    "source_date",
    "observed_at",
    "observed_date",
    "event_date",
)
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_temporal_lag_csv(
    edges: Iterable[KnowledgeEdge],
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write signed day lags between connected units when dates are available."""
    edge_list = list(edges)
    unit_list = list(units)
    rows = _lag_rows(edge_list, unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _lag_rows(edges: list[KnowledgeEdge], units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    unit_by_id = {_field_value(unit.id): unit for unit in units}
    rows: list[dict[str, str | int]] = []
    for edge in edges:
        from_unit_id = _field_value(edge.from_unit_id)
        to_unit_id = _field_value(edge.to_unit_id)
        from_date = _unit_date(unit_by_id.get(from_unit_id))
        to_date = _unit_date(unit_by_id.get(to_unit_id))
        lag_days = (to_date - from_date).days if from_date is not None and to_date is not None else None
        rows.append(
            {
                "edge_id": _field_value(edge.id),
                "relation": _field_value(edge.relation),
                "from_unit_id": from_unit_id,
                "to_unit_id": to_unit_id,
                "from_date": from_date.isoformat() if from_date is not None else "",
                "to_date": to_date.isoformat() if to_date is not None else "",
                "lag_days": lag_days if lag_days is not None else "",
                "lag_direction": _lag_direction(lag_days),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["lag_direction"]),
            _sort_key(row["edge_id"]),
            _sort_key(row["from_unit_id"]),
            _sort_key(row["to_unit_id"]),
        ),
    )


def _unit_date(unit: KnowledgeUnit | None) -> date | None:
    if unit is None:
        return None
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    for key in _METADATA_DATE_KEYS:
        parsed = _date_value(metadata.get(key))
        if parsed is not None:
            return parsed
    for field in _UNIT_DATE_FIELDS:
        parsed = _date_value(getattr(unit, field, None))
        if parsed is not None:
            return parsed
    return None


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


def _lag_direction(lag_days: int | None) -> str:
    if lag_days is None:
        return "missing_date"
    if lag_days < 0:
        return "after"
    if lag_days > 0:
        return "before"
    return "same"


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
