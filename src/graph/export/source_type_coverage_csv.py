"""CSV export for source type coverage."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "source_type",
    "source_count",
    "unit_count",
    "edge_count",
    "average_confidence",
    "missing_metadata_count",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_type_coverage_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge] | None = None,
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write graph coverage grouped by source entity type."""
    unit_list = list(units)
    edge_list = list(edges or [])
    rows = _coverage_rows(unit_list, edge_list)
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
        "source_type_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _coverage_rows(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[_unit_type(unit)].append(unit)

    edge_counts = _edge_counts(edges, units)
    rows: list[dict[str, str | int]] = []
    for source_type in sorted(groups, key=_sort_key):
        type_units = groups[source_type]
        confidences = [
            value for value in (_confidence_value(unit.confidence) for unit in type_units) if value is not None
        ]
        rows.append(
            {
                "source_type": source_type,
                "source_count": len({_unit_source(unit) for unit in type_units}),
                "unit_count": len(type_units),
                "edge_count": edge_counts[source_type],
                "average_confidence": _decimal(sum(confidences) / len(confidences)) if confidences else "",
                "missing_metadata_count": sum(1 for unit in type_units if not _metadata_keys(unit)),
            }
        )
    return rows


def _edge_counts(edges: list[KnowledgeEdge], units: list[KnowledgeUnit]) -> Counter[str]:
    unit_types: dict[str, str] = {}
    for unit in units:
        source_type = _unit_type(unit)
        for key in (_unit_id(unit), _inline_text(unit.source_id)):
            if key:
                unit_types[key] = source_type

    counts: Counter[str] = Counter()
    for edge in edges:
        source_types = {
            unit_types.get(_inline_text(edge.from_unit_id), ""),
            unit_types.get(_inline_text(edge.to_unit_id), ""),
        }
        for source_type in source_types:
            if source_type:
                counts[source_type] += 1
    return counts


def _metadata_keys(unit: KnowledgeUnit) -> list[str]:
    metadata = unit.metadata
    if not isinstance(metadata, Mapping):
        return []
    return [_inline_text(key) for key in metadata if _inline_text(key)]


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _confidence_value(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id) or _inline_text(unit.source_id)


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or "Unknown"


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
