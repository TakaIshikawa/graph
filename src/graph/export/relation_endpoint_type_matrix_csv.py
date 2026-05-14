"""CSV export for relation endpoint entity type patterns."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "source_entity_type",
    "relation",
    "target_entity_type",
    "edge_count",
    "average_weight",
    "average_confidence",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_endpoint_type_matrix_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write relation counts by endpoint entity type as deterministic CSV."""
    unit_list = list(units)
    edge_list = list(edges)
    rows = _matrix_rows(unit_list, edge_list)
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


def _matrix_rows(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
) -> list[dict[str, str | int]]:
    units_by_id = {_field_value(unit.id): unit for unit in units if _field_value(unit.id)}
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"edge_count": 0, "weights": [], "confidences": []}
    )

    for edge in edges:
        source_type = _endpoint_type(units_by_id.get(_field_value(edge.from_unit_id)))
        target_type = _endpoint_type(units_by_id.get(_field_value(edge.to_unit_id)))
        relation = _field_value(edge.relation) or "Unknown"
        group = groups[(source_type, relation, target_type)]
        group["edge_count"] += 1
        weight = _number(getattr(edge, "weight", None))
        if weight is not None:
            group["weights"].append(weight)
        confidence = _edge_confidence(edge)
        if confidence is not None:
            group["confidences"].append(confidence)

    rows: list[dict[str, str | int]] = []
    for source_type, relation, target_type in sorted(
        groups,
        key=lambda key: (_sort_key(key[0]), _sort_key(key[1]), _sort_key(key[2])),
    ):
        group = groups[(source_type, relation, target_type)]
        rows.append(
            {
                "source_entity_type": source_type,
                "relation": relation,
                "target_entity_type": target_type,
                "edge_count": group["edge_count"],
                "average_weight": _average(group["weights"]),
                "average_confidence": _average(group["confidences"]),
            }
        )
    return rows


def _endpoint_type(unit: KnowledgeUnit | None) -> str:
    if unit is None:
        return "Unknown"
    return _field_value(unit.source_entity_type) or "Unknown"


def _edge_confidence(edge: KnowledgeEdge) -> float | None:
    value = _number(getattr(edge, "confidence", None))
    if value is not None:
        return value
    metadata = edge.metadata if isinstance(edge.metadata, Mapping) else {}
    return _number(metadata.get("confidence"))


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _average(values: list[float]) -> str:
    if not values:
        return ""
    return f"{sum(values) / len(values):.2f}"


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
