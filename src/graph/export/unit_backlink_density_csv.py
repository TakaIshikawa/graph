"""CSV export for inbound and outbound unit relation density."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = ["unit_id", "title", "source", "inbound_count", "outbound_count", "total_degree", "relation_types"]
_SOURCE_KEYS = ("from_unit_id", "source_unit_id", "source_id", "from_id", "source", "from")
_TARGET_KEYS = ("to_unit_id", "target_unit_id", "target_id", "to_id", "target", "to")
_RELATION_KEYS = ("relation", "relation_type", "type")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_backlink_density_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    relations: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write relation degree density rows for each unit."""
    unit_list = list(units)
    relation_list = list(relations)
    rows = _density_rows(unit_list, relation_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "relation_count": len(relation_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _density_rows(
    units: list[KnowledgeUnit | Mapping[str, Any]],
    relations: list[KnowledgeEdge | Mapping[str, Any]],
) -> list[dict[str, str | int]]:
    unit_ids = {_unit_id(unit) for unit in units if _unit_id(unit)}
    inbound: Counter[str] = Counter()
    outbound: Counter[str] = Counter()
    relation_types: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        source_id = _edge_endpoint(relation, _SOURCE_KEYS)
        target_id = _edge_endpoint(relation, _TARGET_KEYS)
        relation_type = _relation_type(relation)
        if source_id in unit_ids:
            outbound[source_id] += 1
            if relation_type:
                relation_types[source_id].add(relation_type)
        if target_id in unit_ids:
            inbound[target_id] += 1
            if relation_type:
                relation_types[target_id].add(relation_type)

    rows: list[dict[str, str | int]] = []
    for unit in units:
        unit_id = _unit_id(unit)
        in_count = inbound[unit_id]
        out_count = outbound[unit_id]
        rows.append(
            {
                "unit_id": unit_id,
                "title": _field_value(_get(unit, "title")),
                "source": _field_value(_get(unit, "source_project")) or "Unknown",
                "inbound_count": in_count,
                "outbound_count": out_count,
                "total_degree": in_count + out_count,
                "relation_types": _joined_unique(relation_types[unit_id]),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["title"])))


def _edge_endpoint(edge: KnowledgeEdge | Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _endpoint_text(_get(edge, key))
        if text:
            return text
    return ""


def _endpoint_text(value: object) -> str:
    if isinstance(value, Mapping):
        return _field_value(value.get("id")) or _field_value(value.get("unit_id"))
    object_id = _field_value(getattr(value, "id", None)) or _field_value(getattr(value, "unit_id", None))
    return object_id or _field_value(value)


def _relation_type(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    for key in _RELATION_KEYS:
        value = _field_value(_get(edge, key))
        if value:
            return value
    return ""


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _joined_unique(values: Iterable[object]) -> str:
    return "; ".join(sorted({_field_value(value) for value in values if _field_value(value)}, key=_sort_key))


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
