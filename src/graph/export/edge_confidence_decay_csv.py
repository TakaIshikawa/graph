"""CSV export for edge confidence decay scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, parse_datetime, render_csv, sort_key, write_csv

_FIELDNAMES = ["edge_id", "from_unit_id", "to_unit_id", "confidence", "age_days", "confidence_decay_score"]
_DATE_KEYS = ("updated_at", "created_at", "ingested_at", "timestamp", "date", "observed_at")


def export_edge_confidence_decay_csv(
    edges: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    *,
    reference_date: datetime | str | None = None,
) -> str | dict[str, Any]:
    """Return or write confidence decay rows for edges."""
    edge_list = list(edges)
    ref = parse_datetime(reference_date) or datetime.now(timezone.utc)
    rows = [_row(edge, ref) for edge in edge_list]
    rows.sort(key=lambda row: sort_key(row["edge_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "edge_count": len(edge_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(edge: Mapping[str, Any] | object, reference_date: datetime) -> dict[str, str]:
    confidence = _confidence(edge)
    age = _age_days(edge, reference_date)
    score = confidence / (1 + max(age, 0) / 365)
    return {
        "edge_id": edge_id(edge),
        "from_unit_id": field_value(get(edge, "from_unit_id") or get(edge, "source_id") or get(edge, "from_id")),
        "to_unit_id": field_value(get(edge, "to_unit_id") or get(edge, "target_id") or get(edge, "to_id")),
        "confidence": f"{confidence:.2f}",
        "age_days": str(max(age, 0)),
        "confidence_decay_score": f"{score:.2f}",
    }


def _confidence(edge: Mapping[str, Any] | object) -> float:
    for value in (get(edge, "confidence"), metadata(edge).get("confidence"), get(edge, "weight"), metadata(edge).get("weight")):
        try:
            if field_value(value):
                return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return 1.0


def _age_days(edge: Mapping[str, Any] | object, reference_date: datetime) -> int:
    for key in _DATE_KEYS:
        parsed = parse_datetime(get(edge, key)) or parse_datetime(metadata(edge).get(key))
        if parsed:
            return (reference_date - parsed).days
    return 0
