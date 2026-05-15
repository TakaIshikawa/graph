"""CSV export for relation provenance matrix summaries."""

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
    "relation",
    "source_project",
    "source_entity_type",
    "edge_count",
    "unique_from_units",
    "unique_to_units",
    "average_weight",
    "average_confidence",
]
_UNKNOWN = "Unknown"
_WHITESPACE_RE = re.compile(r"\s+")


def export_relation_provenance_matrix_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write provenance summaries grouped by relation and source metadata."""
    edge_list = list(edges)
    rows = _matrix_rows(edge_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _matrix_rows(edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], list[KnowledgeEdge]] = defaultdict(list)
    for edge in edges:
        metadata = edge.metadata if isinstance(edge.metadata, Mapping) else {}
        groups[
            (
                _field_value(edge.relation) or _UNKNOWN,
                _field_value(metadata.get("source_project")) or _UNKNOWN,
                _field_value(metadata.get("source_entity_type")) or _UNKNOWN,
            )
        ].append(edge)

    rows: list[dict[str, str | int]] = []
    for (relation, source_project, source_entity_type), group_edges in groups.items():
        rows.append(
            {
                "relation": relation,
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "edge_count": len(group_edges),
                "unique_from_units": len({_field_value(edge.from_unit_id) for edge in group_edges}),
                "unique_to_units": len({_field_value(edge.to_unit_id) for edge in group_edges}),
                "average_weight": _average(_number(edge.weight) for edge in group_edges),
                "average_confidence": _average(_number(_metadata(edge).get("confidence")) for edge in group_edges),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["relation"]),
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
        ),
    )


def _metadata(edge: KnowledgeEdge) -> Mapping[str, Any]:
    return edge.metadata if isinstance(edge.metadata, Mapping) else {}


def _average(values: Iterable[float | None]) -> str:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return ""
    return f"{sum(numbers) / len(numbers):.2f}"


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


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
