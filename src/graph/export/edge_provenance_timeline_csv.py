"""CSV export for edge provenance timeline rows."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "edge_id",
    "from_unit_id",
    "to_unit_id",
    "relation",
    "source",
    "source_project",
    "source_entity_type",
    "provenance_date",
    "provenance_key",
    "weight",
    "metadata_key_count",
]
_METADATA_DATE_KEYS = ("observed_at", "observed_date", "source_date", "date", "published_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_provenance_timeline_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one provenance timeline row per edge."""
    edge_list = list(edges)
    rows = _timeline_rows(edge_list)
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


def _timeline_rows(edges: list[KnowledgeEdge]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for edge in edges:
        provenance_date, provenance_key = _provenance(edge)
        metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
        rows.append(
            {
                "edge_id": _field_value(edge.id),
                "from_unit_id": _field_value(edge.from_unit_id),
                "to_unit_id": _field_value(edge.to_unit_id),
                "relation": _field_value(edge.relation) or "Unknown",
                "source": _field_value(edge.source) or "Unknown",
                "source_project": _field_value(metadata.get("source_project")) or "Unknown",
                "source_entity_type": _field_value(metadata.get("source_entity_type")) or "Unknown",
                "provenance_date": provenance_date.isoformat() if provenance_date else "",
                "provenance_key": provenance_key,
                "weight": _decimal(_weight(edge.weight)),
                "metadata_key_count": _metadata_key_count(metadata),
            }
        )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["provenance_date"]),
            _sort_key(row["edge_id"]),
            _sort_key(row["from_unit_id"]),
            _sort_key(row["to_unit_id"]),
        ),
    )


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _provenance(edge: KnowledgeEdge) -> tuple[date | None, str]:
    metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
    for key in _METADATA_DATE_KEYS:
        value = _date_value(metadata.get(key))
        if value is not None:
            return value, key

    created_at = _date_value(getattr(edge, "created_at", None))
    if created_at is not None:
        return created_at, "created_at"
    return None, ""


def _date_value(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _inline_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _metadata_key_count(metadata: dict) -> int:
    return len([key for key in metadata if _field_value(key)])


def _weight(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


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
