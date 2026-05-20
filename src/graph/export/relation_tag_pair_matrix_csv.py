"""CSV export for relation counts between source and target unit tags."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from itertools import product
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = ["relation", "source", "from_tag", "to_tag", "edge_count", "total_weight", "sample_edges"]
_UNTAGGED = "Untagged"
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_tag_pair_matrix_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    include_untagged: bool = True,
) -> str | dict[str, Any]:
    """Return or write relation counts for from-tag/to-tag combinations."""
    edge_list = list(edges)
    unit_list = list(units)
    rows = _matrix_rows(edge_list, unit_list, include_untagged)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "edge_count": len(edge_list), "unit_count": len(unit_list), "rows_exported": len(rows), "include_untagged": include_untagged, "bytes_written": output_path.stat().st_size}


def _matrix_rows(edges: list[KnowledgeEdge | Mapping[str, Any]], units: list[KnowledgeUnit | Mapping[str, Any]], include_untagged: bool) -> list[dict[str, str | int]]:
    lookup: dict[str, KnowledgeUnit | Mapping[str, Any]] = {}
    for unit in units:
        for key in (_field_value(_get(unit, "id")), _field_value(_get(unit, "source_id"))):
            if key:
                lookup[key] = unit
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(lambda: {"edge_count": 0, "weight": 0.0, "samples": []})
    for edge in edges:
        from_unit = lookup.get(_source_id(edge))
        to_unit = lookup.get(_target_id(edge))
        if from_unit is None or to_unit is None:
            continue
        from_tags = _unit_tags(from_unit, include_untagged)
        to_tags = _unit_tags(to_unit, include_untagged)
        if not from_tags or not to_tags:
            continue
        for from_tag, to_tag in product(from_tags, to_tags):
            group = groups[(_relation(edge), _source(edge), from_tag, to_tag)]
            group["edge_count"] += 1
            group["weight"] += _weight(edge)
            if len(group["samples"]) < 5:
                group["samples"].append(_edge_id(edge))
    rows: list[dict[str, str | int]] = []
    for (relation, source, from_tag, to_tag), values in groups.items():
        rows.append({"relation": relation, "source": source, "from_tag": from_tag, "to_tag": to_tag, "edge_count": values["edge_count"], "total_weight": _decimal(values["weight"]), "sample_edges": "; ".join(sorted(values["samples"], key=_sort_key))})
    return sorted(rows, key=lambda row: (_sort_key(row["relation"]), _sort_key(row["source"]), _sort_key(row["from_tag"]), _sort_key(row["to_tag"]), -int(row["edge_count"])))


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any], include_untagged: bool) -> list[str]:
    values = _get(unit, "tags", [])
    tags = {_field_value(tag) for tag in values} if isinstance(values, Iterable) and not isinstance(values, str | bytes | Mapping) else ({_field_value(values)} if _field_value(values) else set())
    if not tags:
        metadata_tags = _metadata(unit).get("tags")
        if isinstance(metadata_tags, Iterable) and not isinstance(metadata_tags, str | bytes | Mapping):
            tags = {_field_value(tag) for tag in metadata_tags if _field_value(tag)}
        elif _field_value(metadata_tags):
            tags = {_field_value(metadata_tags)}
    if tags:
        return sorted(tags, key=_sort_key)
    return [_UNTAGGED] if include_untagged else []


def _source_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("from_unit_id", "source_id", "from_id", "source")))


def _target_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("to_unit_id", "target_id", "to_id", "target")))


def _edge_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_get(edge, "id")) or f"{_source_id(edge)}->{_target_id(edge)}"


def _relation(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("relation", "relation_type", "type"))) or "Unknown"


def _source(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("source", "source_project", "provider"))) or "Unknown"


def _weight(edge: KnowledgeEdge | Mapping[str, Any]) -> float:
    try:
        return float(_get(edge, "weight", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _first_value(value: object, keys: tuple[str, ...], default: object = "") -> object:
    for key in keys:
        item = _get(value, key, None)
        if item is not None:
            return item
    metadata = _get(value, "metadata", {})
    if isinstance(metadata, Mapping):
        for key in keys:
            if key in metadata:
                return metadata[key]
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
