"""CSV export for relation self loops."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import edge_id, field_value, get, metadata, render_csv, sort_key, write_csv

_FIELDNAMES = ["relation_id", "relation_type", "endpoint_id", "source_label", "target_label", "metadata_source"]
_RELATION_KEYS = ("relation", "relation_type", "type", "predicate")
_SOURCE_KEYS = ("from_unit_id", "source_unit_id", "source_id", "from_id", "source")
_TARGET_KEYS = ("to_unit_id", "target_unit_id", "target_id", "to_id", "target")
_SOURCE_LABEL_KEYS = ("source_label", "from_label")
_TARGET_LABEL_KEYS = ("target_label", "to_label")
_METADATA_SOURCE_KEYS = ("metadata_source", "source", "edge_source", "source_project")


def export_relation_self_loop_csv(
    relations: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    relation_list = list(relations)
    rows: list[dict[str, str]] = []
    for index, relation in enumerate(relation_list):
        meta = metadata(relation)
        source_id = _value(relation, meta, _SOURCE_KEYS)
        target_id = _value(relation, meta, _TARGET_KEYS)
        if not source_id or not target_id or source_id != target_id:
            continue
        rows.append(
            {
                "relation_id": edge_id(relation) or str(index),
                "relation_type": _value(relation, meta, _RELATION_KEYS) or "unknown",
                "endpoint_id": source_id,
                "source_label": _value(relation, meta, _SOURCE_LABEL_KEYS),
                "target_label": _value(relation, meta, _TARGET_LABEL_KEYS),
                "metadata_source": _value(relation, meta, _METADATA_SOURCE_KEYS),
            }
        )
    rows.sort(key=lambda row: (sort_key(row["relation_type"]), sort_key(row["endpoint_id"]), sort_key(row["relation_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "relation_count": len(relation_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _value(item: Any, meta: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = field_value(get(item, key))
        if value:
            return value
        value = field_value(meta.get(key))
        if value:
            return value
    return ""
