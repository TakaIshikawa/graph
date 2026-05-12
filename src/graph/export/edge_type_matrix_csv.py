"""CSV export for edge relation counts by endpoint unit type."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = ["relation", "source_type", "target_type", "count"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_type_matrix_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
    *,
    include_zeroes: bool = False,
) -> str | dict[str, Any]:
    """Return or write relation counts by source and target unit type."""
    unit_list = list(units)
    edge_list = list(edges)
    rows, skipped_edges = _matrix_rows(unit_list, edge_list, include_zeroes=include_zeroes)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "units_scanned": len(unit_list),
        "edges_scanned": len(edge_list),
        "rows_exported": len(rows),
        "skipped_edges": skipped_edges,
        "include_zeroes": include_zeroes,
        "bytes_written": output_path.stat().st_size,
    }


def _matrix_rows(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
    *,
    include_zeroes: bool,
) -> tuple[list[dict[str, int | str]], int]:
    units_by_id = {_unit_id(unit): unit for unit in units}
    counts: Counter[tuple[str, str, str]] = Counter()
    relations: set[str] = set()
    types = {_unit_type(unit) for unit in units}
    skipped_edges = 0

    for edge in edges:
        relation = _field_value(edge.relation)
        relations.add(relation)
        source = units_by_id.get(_inline_text(edge.from_unit_id))
        target = units_by_id.get(_inline_text(edge.to_unit_id))
        if source is None or target is None:
            skipped_edges += 1
            continue
        counts[(relation, _unit_type(source), _unit_type(target))] += 1

    keys = set(counts)
    if include_zeroes:
        keys.update((relation, source_type, target_type) for relation in relations for source_type in types for target_type in types)

    rows = [
        {
            "relation": relation,
            "source_type": source_type,
            "target_type": target_type,
            "count": counts.get((relation, source_type, target_type), 0),
        }
        for relation, source_type, target_type in sorted(keys, key=lambda key: tuple(_sort_key(part) for part in key))
    ]
    return rows, skipped_edges


def _render_csv(rows: list[dict[str, int | str]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id or unit.source_id)


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or _field_value(unit.content_type) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
