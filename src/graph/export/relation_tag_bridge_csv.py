"""CSV export for relation tag overlap bridge checks."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "relation",
    "source_unit_id",
    "target_unit_id",
    "source_tag_count",
    "target_tag_count",
    "shared_tag_count",
    "bridge_bucket",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_tag_bridge_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write tag-overlap buckets for edges."""
    unit_list = list(units)
    edge_list = list(edges)
    rows = _bridge_rows(unit_list, edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "edge_count": len(edge_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _bridge_rows(units: list[KnowledgeUnit], edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    tag_lookup = {_field_value(unit.id): _unit_tags(unit) for unit in units}
    rows: list[dict[str, str | int]] = []
    for edge in edges:
        source_id = _field_value(edge.from_unit_id)
        target_id = _field_value(edge.to_unit_id)
        source_tags = tag_lookup.get(source_id, set())
        target_tags = tag_lookup.get(target_id, set())
        shared_tags = source_tags & target_tags
        rows.append(
            {
                "relation": _field_value(edge.relation),
                "source_unit_id": source_id,
                "target_unit_id": target_id,
                "source_tag_count": len(source_tags),
                "target_tag_count": len(target_tags),
                "shared_tag_count": len(shared_tags),
                "bridge_bucket": _bridge_bucket(source_tags, target_tags, shared_tags),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["relation"]),
            _sort_key(row["source_unit_id"]),
            _sort_key(row["target_unit_id"]),
        ),
    )


def _unit_tags(unit: KnowledgeUnit) -> set[str]:
    values = list(unit.tags or [])
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    if not values:
        values.extend(_iter_values(metadata.get("tags")))
    tags: set[str] = set()
    for value in values:
        tag = _inline_text(value).casefold()
        if tag:
            tags.add(tag)
    return tags


def _iter_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return list(value)
    if value is None:
        return []
    return [value]


def _bridge_bucket(source_tags: set[str], target_tags: set[str], shared_tags: set[str]) -> str:
    if not source_tags and not target_tags:
        return "no_tags"
    if not shared_tags:
        return "disjoint"
    if source_tags == target_tags:
        return "same_tags"
    return "partial_overlap"


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
