"""CSV export for relation polarity conflicts between endpoints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, normalized_key, render_csv, sort_key, write_csv

_FIELDNAMES = ["endpoint_pair", "relation_type", "conflicting_polarities", "relation_ids", "conflict_count"]
_POLARITY_KEYS = ("polarity", "sentiment", "stance", "relation_polarity")
_POSITIVE = {"positive", "support", "supports", "supported", "pro", "agree", "agreement", "true"}
_NEGATIVE = {"negative", "contradict", "contradicts", "contradicted", "anti", "against", "disagree", "false"}


def export_relation_polarity_conflict_csv(
    relations: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write relation endpoint groups that contain opposing polarities."""
    relation_list = list(relations)
    rows = _conflict_rows(relation_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "relation_count": len(relation_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _conflict_rows(relations: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any] | object]] = defaultdict(list)
    for relation in relations:
        polarity = _polarity(relation)
        if polarity:
            groups[(_endpoint_pair(relation), _relation_type(relation))].append(relation)
    rows: list[dict[str, str | int]] = []
    for (endpoint_pair, relation_type), group in groups.items():
        polarities = sorted({_polarity(item) for item in group if _polarity(item)}, key=sort_key)
        if "positive" in polarities and "negative" in polarities:
            relation_ids = sorted((edge_id(item) for item in group if edge_id(item)), key=sort_key)
            rows.append(
                {
                    "endpoint_pair": endpoint_pair,
                    "relation_type": relation_type,
                    "conflicting_polarities": "; ".join(polarities),
                    "relation_ids": "; ".join(relation_ids),
                    "conflict_count": len(group),
                }
            )
    return sorted(rows, key=lambda row: (sort_key(row["endpoint_pair"]), sort_key(row["relation_type"])))


def _endpoint_pair(relation: Mapping[str, Any] | object) -> str:
    endpoints = sorted([field_value(get(relation, "from_unit_id") or get(relation, "source_id") or get(relation, "from_id")), field_value(get(relation, "to_unit_id") or get(relation, "target_id") or get(relation, "to_id"))], key=sort_key)
    return " <-> ".join(endpoints)


def _relation_type(relation: Mapping[str, Any] | object) -> str:
    return field_value(get(relation, "relation") or get(relation, "relation_type") or get(relation, "type")) or "unknown"


def _polarity(relation: Mapping[str, Any] | object) -> str:
    for key in _POLARITY_KEYS:
        text = _normalize_polarity(get(relation, key))
        if text:
            return text
    for key, value in metadata(relation).items():
        if normalized_key(key) in _POLARITY_KEYS:
            text = _normalize_polarity(value)
            if text:
                return text
    relation_type = field_value(get(relation, "relation") or get(relation, "relation_type") or get(relation, "type"))
    return _normalize_polarity(relation_type)


def _normalize_polarity(value: object) -> str:
    text = normalized_key(value)
    if text in _POSITIVE:
        return "positive"
    if text in _NEGATIVE:
        return "negative"
    return ""
