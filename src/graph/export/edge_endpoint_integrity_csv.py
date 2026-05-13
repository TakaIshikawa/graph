"""CSV export for edges with unresolved endpoints."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "edge_id",
    "relation",
    "source",
    "from_unit_id",
    "to_unit_id",
    "missing_from_unit",
    "missing_to_unit",
    "issue_count",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_endpoint_integrity_csv(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write rows for edges whose endpoints do not resolve to known units."""
    unit_list = list(units)
    edge_list = list(edges)
    rows = _integrity_rows(unit_list, edge_list)
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
        "problem_edge_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _integrity_rows(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
) -> list[dict[str, str | int]]:
    resolvable_ids = _resolvable_unit_ids(units)
    rows: list[dict[str, str | int]] = []

    for edge in edges:
        from_unit_id = _inline_text(edge.from_unit_id)
        to_unit_id = _inline_text(edge.to_unit_id)
        missing_from = from_unit_id not in resolvable_ids
        missing_to = to_unit_id not in resolvable_ids
        issue_count = int(missing_from) + int(missing_to)
        if issue_count == 0:
            continue

        rows.append(
            {
                "edge_id": _inline_text(edge.id),
                "relation": _field_value(edge.relation),
                "source": _field_value(edge.source),
                "from_unit_id": from_unit_id,
                "to_unit_id": to_unit_id,
                "missing_from_unit": _bool_text(missing_from),
                "missing_to_unit": _bool_text(missing_to),
                "issue_count": issue_count,
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            -int(row["issue_count"]),
            _sort_key(row["relation"]),
            _sort_key(row["edge_id"]),
            _sort_key(row["from_unit_id"]),
            _sort_key(row["to_unit_id"]),
        ),
    )


def _resolvable_unit_ids(units: Iterable[KnowledgeUnit]) -> set[str]:
    ids: set[str] = set()
    for unit in units:
        unit_id = _inline_text(unit.id)
        source_id = _inline_text(unit.source_id)
        if unit_id:
            ids.add(unit_id)
        if source_id:
            ids.add(source_id)
    return ids


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
