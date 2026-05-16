"""CSV export for directed relation reciprocity gaps."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "source_id",
    "target_id",
    "relation_type",
    "weight",
    "reciprocal_relation_type",
    "reciprocal_present",
    "reciprocal_weight",
    "gap_reason",
    "evidence_count",
]
DEFAULT_RECIPROCAL_MAP = {
    "references": "referenced_by",
    "referenced_by": "references",
    "parent": "child",
    "child": "parent",
    "supports": "supported_by",
    "supported_by": "supports",
    "depends_on": "required_by",
    "required_by": "depends_on",
    "blocks": "blocked_by",
    "blocked_by": "blocks",
    "relates_to": "relates_to",
    "related_to": "related_to",
}
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_reciprocity_gaps_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    reciprocal_map: Mapping[str, str] | None = None,
) -> str | dict[str, Any]:
    """Return or write directed relation rows with missing expected reciprocal edges."""
    edge_list = list(edges)
    relation_map = _relation_map(reciprocal_map)
    rows = _gap_rows(edge_list, relation_map)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "gap_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _gap_rows(
    edges: list[KnowledgeEdge | Mapping[str, Any]],
    reciprocal_map: Mapping[str, str],
) -> list[dict[str, str | int]]:
    indexed: dict[tuple[str, str, str], list[KnowledgeEdge | Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        source_id = _source_id(edge)
        target_id = _target_id(edge)
        relation_type = _relation_type(edge)
        if source_id and target_id and relation_type:
            indexed[(source_id, target_id, relation_type)].append(edge)

    rows: list[dict[str, str | int]] = []
    for edge in edges:
        source_id = _source_id(edge)
        target_id = _target_id(edge)
        relation_type = _relation_type(edge)
        if not source_id or not target_id or not relation_type:
            continue
        reciprocal_relation_type = reciprocal_map.get(relation_type, relation_type)
        reverse_edges = indexed.get((target_id, source_id, reciprocal_relation_type), [])
        if reverse_edges:
            continue
        rows.append(
            {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "weight": _decimal(_weight(edge)),
                "reciprocal_relation_type": reciprocal_relation_type,
                "reciprocal_present": "false",
                "reciprocal_weight": "",
                "gap_reason": "missing_reverse_edge",
                "evidence_count": _evidence_count(edge),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source_id"]),
            _sort_key(row["target_id"]),
            _sort_key(row["relation_type"]),
        ),
    )


def _relation_map(values: Mapping[str, str] | None) -> dict[str, str]:
    relation_map = dict(DEFAULT_RECIPROCAL_MAP)
    if values:
        relation_map.update({_field_value(key): _field_value(value) for key, value in values.items()})
    return relation_map


def _source_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("source_id", "from_unit_id", "from_id", "source")))


def _target_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("target_id", "to_unit_id", "to_id", "target")))


def _relation_type(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("relation_type", "relation", "type")))


def _weight(edge: KnowledgeEdge | Mapping[str, Any]) -> float:
    try:
        return float(_first_value(edge, ("weight",), 0.0))
    except (TypeError, ValueError):
        return 0.0


def _evidence_count(edge: KnowledgeEdge | Mapping[str, Any]) -> int:
    metadata = _metadata(edge)
    evidence = _first_present(metadata, ("evidence", "evidence_ids", "citations", "sources"), None)
    if evidence is None:
        evidence = _first_value(edge, ("evidence", "evidence_ids", "citations"), None)
    if isinstance(evidence, str):
        return 1 if _field_value(evidence) else 0
    if isinstance(evidence, Mapping):
        return len(evidence)
    if isinstance(evidence, Iterable):
        return len(list(evidence))
    count = _first_present(metadata, ("evidence_count",), None)
    if count is None:
        count = _first_value(edge, ("evidence_count",), None)
    try:
        return max(0, int(count))
    except (TypeError, ValueError):
        return 0


def _metadata(edge: KnowledgeEdge | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _first_value(edge, ("metadata",), None)
    return metadata if isinstance(metadata, Mapping) else {}


def _first_value(value: object, keys: tuple[str, ...], default: object = "") -> object:
    for key in keys:
        item = _get(value, key, None)
        if item is not None:
            return item
    return default


def _first_present(value: Mapping[str, Any], keys: tuple[str, ...], default: object = None) -> object:
    for key in keys:
        if key in value:
            return value[key]
    return default


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


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


def _decimal(value: float) -> str:
    return f"{value:.2f}"
