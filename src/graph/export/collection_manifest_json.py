"""JSON manifest export for selected graph collections."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_SCHEMA_VERSION = "1.0"
_WHITESPACE_RE = re.compile(r"\s+")


def export_collection_manifest_json(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge] = (),
    path: str | Path | None = None,
    *,
    title: str | None = None,
    include_edges: bool = True,
    generated_at: str | datetime | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic JSON manifest for selected units."""
    unit_list = sorted(list(units), key=_unit_sort_key)
    unit_ids = {_unit_id(unit) for unit in unit_list}
    edge_list = sorted(
        [
            edge
            for edge in edges
            if include_edges
            and _inline_text(edge.from_unit_id) in unit_ids
            and _inline_text(edge.to_unit_id) in unit_ids
        ],
        key=_edge_sort_key,
    )
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": _generated_at(generated_at),
        "unit_count": len(unit_list),
        "edge_count": len(edge_list),
        "sources": dict(sorted(Counter(_unit_source(unit) for unit in unit_list).items(), key=lambda item: _sort_key(item[0]))),
        "tags": dict(sorted(_tag_counts(unit_list).items(), key=lambda item: _sort_key(item[0]))),
        "units": [_unit_record(unit) for unit in unit_list],
        "edges": [_edge_record(edge) for edge in edge_list],
    }
    if title is not None:
        manifest["title"] = _inline_text(title)

    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "edge_count": len(edge_list),
        "bytes_written": output_path.stat().st_size,
    }


def _unit_record(unit: KnowledgeUnit) -> dict[str, Any]:
    metadata = unit.metadata or {}
    record = {
        "id": _unit_id(unit),
        "title": _unit_title(unit),
        "source": _unit_source(unit),
        "source_id": _inline_text(unit.source_id),
        "type": _unit_type(unit),
        "tags": sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key),
    }
    for key in ("created_at", "ingested_at", "updated_at"):
        record[key] = _json_value(getattr(unit, key))
    if metadata:
        record["metadata"] = _json_value(metadata)
    return record


def _edge_record(edge: KnowledgeEdge) -> dict[str, Any]:
    record = {
        "id": _edge_id(edge),
        "from_unit_id": _inline_text(edge.from_unit_id),
        "to_unit_id": _inline_text(edge.to_unit_id),
        "relation": _field_value(edge.relation),
        "source": _field_value(edge.source),
        "weight": edge.weight,
    }
    if edge.metadata:
        record["metadata"] = _json_value(edge.metadata)
    return record


def _tag_counts(units: Iterable[KnowledgeUnit]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for unit in units:
        for tag in {_inline_text(tag) for tag in unit.tags if _inline_text(tag)}:
            counts[tag] += 1
    return counts


def _generated_at(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return _inline_text(value)


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id or unit.source_id)


def _edge_id(edge: KnowledgeEdge) -> str:
    edge_id = _inline_text(edge.id)
    if edge_id:
        return edge_id
    return "|".join((_inline_text(edge.from_unit_id), _inline_text(edge.to_unit_id), _field_value(edge.relation)))


def _unit_title(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata or {}
    for value in (unit.title, metadata.get("title"), metadata.get("name"), metadata.get("label"), unit.source_id, unit.id):
        text = _inline_text(value)
        if text:
            return text
    return "Untitled"


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or _field_value(unit.content_type) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (_unit_title(unit).casefold(), _unit_source(unit).casefold(), _unit_id(unit))


def _edge_sort_key(edge: KnowledgeEdge) -> tuple[str, str, str, str]:
    return (
        _inline_text(edge.from_unit_id),
        _inline_text(edge.to_unit_id),
        _field_value(edge.relation),
        _edge_id(edge),
    )


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]
    return str(value)
