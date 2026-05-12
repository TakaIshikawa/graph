"""CSV export for source freshness summaries."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge, KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "first_seen_date",
    "last_seen_date",
    "observed_date_span_days",
    "unit_count",
    "edge_count",
]
_DATE_FIELDS = ("created_at", "ingested_at", "updated_at")
_METADATA_DATE_KEYS = ("observed_at", "observed_date", "source_date", "date", "published_at")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_freshness_summary_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    edges: Iterable[KnowledgeEdge] | None = None,
) -> str | dict[str, Any]:
    """Return or write source/type freshness statistics as deterministic CSV."""
    unit_list = list(units)
    edge_list = list(edges or [])
    rows = _summary_rows(unit_list, edge_list)
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


def _summary_rows(
    units: list[KnowledgeUnit],
    edges: list[KnowledgeEdge],
) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"unit_count": 0, "edge_count": 0, "dates": []}
    )

    for unit in units:
        key = (_unit_source(unit), _unit_source_type(unit))
        groups[key]["unit_count"] += 1
        groups[key]["dates"].extend(_unit_dates(unit))

    for edge in edges:
        key = _edge_source_key(edge)
        if key is None:
            continue
        groups[key]["edge_count"] += 1
        groups[key]["dates"].extend(_edge_dates(edge))

    rows: list[dict[str, str | int]] = []
    for source_project, source_entity_type in sorted(
        groups,
        key=lambda key: (_sort_key(key[0]), _sort_key(key[1])),
    ):
        values = sorted(groups[(source_project, source_entity_type)]["dates"])
        first_seen = values[0] if values else None
        last_seen = values[-1] if values else None
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "first_seen_date": first_seen.isoformat() if first_seen else "",
                "last_seen_date": last_seen.isoformat() if last_seen else "",
                "observed_date_span_days": (last_seen - first_seen).days if first_seen and last_seen else "",
                "unit_count": groups[(source_project, source_entity_type)]["unit_count"],
                "edge_count": groups[(source_project, source_entity_type)]["edge_count"],
            }
        )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_dates(unit: KnowledgeUnit) -> list[date]:
    values = [_date_value(getattr(unit, field, None)) for field in _DATE_FIELDS]
    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    values.extend(_date_value(metadata.get(key)) for key in _METADATA_DATE_KEYS)
    return [value for value in values if value is not None]


def _edge_dates(edge: KnowledgeEdge) -> list[date]:
    metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
    metadata_dates = [_date_value(metadata.get(key)) for key in _METADATA_DATE_KEYS]
    values = [value for value in metadata_dates if value is not None]
    if values:
        return values
    created_at = _date_value(getattr(edge, "created_at", None))
    return [created_at] if created_at is not None else []


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


def _edge_source_key(edge: KnowledgeEdge) -> tuple[str, str] | None:
    metadata = edge.metadata if isinstance(edge.metadata, dict) else {}
    source_project = _field_value(metadata.get("source_project"))
    source_entity_type = _field_value(metadata.get("source_entity_type"))
    if not source_project and not source_entity_type:
        return None
    return (source_project or "Unknown", source_entity_type or "Unknown")


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_source_type(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_entity_type) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
