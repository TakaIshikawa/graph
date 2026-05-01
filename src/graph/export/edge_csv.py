"""CSV export helpers for graph edges."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_BASE_FIELDNAMES = [
    "id",
    "from_unit_id",
    "to_unit_id",
    "relation",
    "weight",
    "source",
    "created_at",
]
_METADATA_FIELDNAME = "metadata_json"


def export_edges_to_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
    *,
    include_metadata: bool = True,
) -> dict:
    """Write edges to CSV with stable headers and deterministic row ordering."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_edges = list(edges)
    exported_edges = sorted(
        all_edges,
        key=lambda edge: (
            _text(edge.from_unit_id),
            _text(edge.to_unit_id),
            _enum_value(edge.relation),
            _text(edge.id),
        ),
    )
    fieldnames = [*_BASE_FIELDNAMES]
    if include_metadata:
        fieldnames.append(_METADATA_FIELDNAME)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for edge in exported_edges:
            writer.writerow(_edge_row(edge, include_metadata=include_metadata))

    bytes_written = output_path.stat().st_size
    return {
        "path": str(output_path),
        "edges_scanned": len(all_edges),
        "edges_exported": len(exported_edges),
        "bytes_written": bytes_written,
    }


def _edge_row(edge: KnowledgeEdge, *, include_metadata: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": _text(edge.id),
        "from_unit_id": _text(edge.from_unit_id),
        "to_unit_id": _text(edge.to_unit_id),
        "relation": _enum_value(edge.relation),
        "weight": edge.weight,
        "source": _enum_value(edge.source),
        "created_at": _datetime_text(edge.created_at),
    }
    if include_metadata:
        row[_METADATA_FIELDNAME] = json.dumps(
            edge.metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    return row


def _enum_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _datetime_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return _text(value)


def _text(value: object) -> str:
    return str(value or "")
