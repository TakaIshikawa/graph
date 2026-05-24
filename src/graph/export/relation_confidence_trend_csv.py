"""CSV export for relation confidence trends by month."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, render_csv, sort_key, write_csv

_FIELDNAMES = ["period", "relation_type", "relation_count", "min_confidence", "max_confidence", "average_confidence", "low_confidence_count"]
_DATE_KEYS = ("date", "timestamp", "created_at", "updated_at", "observed_at", "event_date", "relation_date")
_CONFIDENCE_KEYS = ("confidence", "score", "weight", "confidence_score")


def export_relation_confidence_trend_csv(
    relations: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write monthly confidence statistics grouped by relation type."""
    relation_list = list(relations)
    rows = _rows(relation_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "relation_count": len(relation_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(relations: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[float | None]] = defaultdict(list)
    for relation in relations:
        groups[(_period(relation), _relation_type(relation))].append(_confidence(relation))
    rows: list[dict[str, str | int]] = []
    for period, relation_type in sorted(groups, key=lambda key: (sort_key(key[0]), sort_key(key[1]))):
        values = groups[(period, relation_type)]
        confidences = [value for value in values if value is not None]
        rows.append(
            {
                "period": period,
                "relation_type": relation_type,
                "relation_count": len(values),
                "min_confidence": _decimal(min(confidences)) if confidences else "",
                "max_confidence": _decimal(max(confidences)) if confidences else "",
                "average_confidence": _decimal(sum(confidences) / len(confidences)) if confidences else "",
                "low_confidence_count": sum(1 for value in confidences if value < 0.5),
            }
        )
    return rows


def _period(relation: Mapping[str, Any] | object) -> str:
    data = metadata(relation)
    for key in _DATE_KEYS:
        parsed = parse_datetime(get(relation, key)) or parse_datetime(data.get(key))
        if parsed:
            return parsed.strftime("%Y-%m")
    return "unknown"


def _relation_type(relation: Mapping[str, Any] | object) -> str:
    return field_value(get(relation, "relation")) or field_value(get(relation, "relation_type")) or "Unknown"


def _confidence(relation: Mapping[str, Any] | object) -> float | None:
    data = metadata(relation)
    for key in _CONFIDENCE_KEYS:
        value = get(relation, key)
        if _is_number(value):
            return float(value)
        value = data.get(key)
        if _is_number(value):
            return float(value)
    return None


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
