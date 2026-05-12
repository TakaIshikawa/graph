"""CSV export for relation temporal lag diagnostics."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "edge_id",
    "from_unit_id",
    "to_unit_id",
    "relation",
    "edge_created_date",
    "evidence_date",
    "lag_days",
    "lag_bucket",
    "source_project",
    "source_entity_type",
]
_METADATA_DATE_KEYS = ("observed_at", "observed_date", "source_date", "date", "published_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_temporal_lag_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one edge timing diagnostic row per relation edge."""
    edge_list = list(edges)
    rows = _lag_rows(edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _lag_rows(edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for edge in edges:
        metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
        edge_created_date = _date_value(getattr(edge, "created_at", None))
        evidence_date = _evidence_date(metadata)
        lag_days = (edge_created_date - evidence_date).days if edge_created_date and evidence_date else None
        rows.append(
            {
                "edge_id": _field_value(edge.id),
                "from_unit_id": _field_value(edge.from_unit_id),
                "to_unit_id": _field_value(edge.to_unit_id),
                "relation": _field_value(edge.relation) or "Unknown",
                "edge_created_date": edge_created_date.isoformat() if edge_created_date else "",
                "evidence_date": evidence_date.isoformat() if evidence_date else "",
                "lag_days": lag_days if lag_days is not None else "",
                "lag_bucket": _lag_bucket(lag_days),
                "source_project": _field_value(metadata.get("source_project")) or "Unknown",
                "source_entity_type": _field_value(metadata.get("source_entity_type")) or "Unknown",
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["relation"]),
            _sort_key(row["edge_id"]),
            _sort_key(row["from_unit_id"]),
            _sort_key(row["to_unit_id"]),
        ),
    )


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _evidence_date(metadata: dict) -> date | None:
    for key in _METADATA_DATE_KEYS:
        value = _date_value(metadata.get(key))
        if value is not None:
            return value
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


def _lag_bucket(lag_days: int | None) -> str:
    if lag_days is None:
        return "undated"
    absolute_days = abs(lag_days)
    if absolute_days == 0:
        return "same_day"
    if absolute_days <= 7:
        return "week"
    if absolute_days <= 31:
        return "month"
    if absolute_days <= 92:
        return "quarter"
    if absolute_days <= 366:
        return "year"
    return "over_year"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
