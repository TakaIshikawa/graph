"""CSV export for high-degree relation endpoints."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = ["relation", "source", "unit_id", "inbound_count", "outbound_count", "total_count", "distinct_neighbor_count", "top_neighbor_ids"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_hub_endpoints_csv(
    edges: Iterable[KnowledgeEdge | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    min_total_count: int = 2,
    neighbor_limit: int = 5,
) -> str | dict[str, Any]:
    """Return or write high-degree endpoints per relation/source."""
    min_total_count = _positive_int(min_total_count, "min_total_count")
    neighbor_limit = _positive_int(neighbor_limit, "neighbor_limit")
    edge_list = list(edges)
    rows = _hub_rows(edge_list, min_total_count, neighbor_limit)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "edge_count": len(edge_list), "rows_exported": len(rows), "min_total_count": min_total_count, "neighbor_limit": neighbor_limit, "bytes_written": output_path.stat().st_size}


def _hub_rows(edges: list[KnowledgeEdge | Mapping[str, Any]], min_total_count: int, neighbor_limit: int) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"in": 0, "out": 0, "neighbors": Counter()})
    for edge in edges:
        from_id = _source_id(edge)
        to_id = _target_id(edge)
        if not from_id or not to_id:
            continue
        relation = _relation(edge)
        source = _source(edge)
        groups[(relation, source, from_id)]["out"] += 1
        groups[(relation, source, from_id)]["neighbors"][to_id] += 1
        groups[(relation, source, to_id)]["in"] += 1
        groups[(relation, source, to_id)]["neighbors"][from_id] += 1
    rows: list[dict[str, str | int]] = []
    for (relation, source, unit_id), values in groups.items():
        total = values["in"] + values["out"]
        if total < min_total_count:
            continue
        top_neighbors = [neighbor for neighbor, _ in sorted(values["neighbors"].items(), key=lambda item: (-item[1], _sort_key(item[0])))[:neighbor_limit]]
        rows.append({"relation": relation, "source": source, "unit_id": unit_id, "inbound_count": values["in"], "outbound_count": values["out"], "total_count": total, "distinct_neighbor_count": len(values["neighbors"]), "top_neighbor_ids": "; ".join(top_neighbors)})
    return sorted(rows, key=lambda row: (-int(row["total_count"]), _sort_key(row["relation"]), _sort_key(row["source"]), _sort_key(row["unit_id"])))


def _source_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("from_unit_id", "source_id", "from_id", "source")))


def _target_id(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("to_unit_id", "target_id", "to_id", "target")))


def _relation(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("relation", "relation_type", "type"))) or "Unknown"


def _source(edge: KnowledgeEdge | Mapping[str, Any]) -> str:
    return _field_value(_first_value(edge, ("source", "source_project", "provider"))) or "Unknown"


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


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


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
