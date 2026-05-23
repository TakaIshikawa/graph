"""CSV export for relation type counts by endpoint unit type pair."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, render_csv, sort_key, unit_id, write_csv
from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = ["relation_type", "source_type", "target_type", "count", "relation_ids"]
_UNKNOWN = "unknown"


def export_relation_type_pair_matrix_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    relations: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write relation counts grouped by relation and endpoint types."""
    unit_list = list(units)
    relation_list = list(relations)
    rows = _matrix_rows(unit_list, relation_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "relation_count": len(relation_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _matrix_rows(units: list[KnowledgeUnit | Mapping[str, Any]], relations: list[KnowledgeEdge | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    units_by_id = {unit_id(unit): unit for unit in units if unit_id(unit)}
    buckets: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"count": 0, "ids": set()})
    for relation in relations:
        source_type = _relation_endpoint_type(relation, units_by_id, "source", "from_unit_id", "source_unit_id")
        target_type = _relation_endpoint_type(relation, units_by_id, "target", "to_unit_id", "target_unit_id")
        relation_type = field_value(get(relation, "relation")) or field_value(get(relation, "relation_type")) or field_value(get(relation, "type")) or _UNKNOWN
        bucket = buckets[(relation_type, source_type, target_type)]
        bucket["count"] += 1
        if edge_id(relation):
            bucket["ids"].add(edge_id(relation))

    rows: list[dict[str, str | int]] = []
    for relation_type, source_type, target_type in sorted(buckets, key=lambda key: (sort_key(key[0]), sort_key(key[1]), sort_key(key[2]))):
        bucket = buckets[(relation_type, source_type, target_type)]
        rows.append(
            {
                "relation_type": relation_type,
                "source_type": source_type,
                "target_type": target_type,
                "count": bucket["count"],
                "relation_ids": "; ".join(sorted(bucket["ids"], key=sort_key)),
            }
        )
    return rows


def _relation_endpoint_type(relation: KnowledgeEdge | Mapping[str, Any], units_by_id: Mapping[str, object], prefix: str, *id_keys: str) -> str:
    explicit = field_value(get(relation, f"{prefix}_type")) or field_value(metadata(relation).get(f"{prefix}_type"))
    if explicit:
        return explicit
    for key in id_keys:
        endpoint_id = field_value(get(relation, key))
        if endpoint_id and endpoint_id in units_by_id:
            return _unit_type(units_by_id[endpoint_id])
    return _UNKNOWN


def _unit_type(unit: object) -> str:
    return (
        field_value(get(unit, "unit_type"))
        or field_value(get(unit, "source_entity_type"))
        or field_value(get(unit, "content_type"))
        or field_value(metadata(unit).get("type"))
        or _UNKNOWN
    )

