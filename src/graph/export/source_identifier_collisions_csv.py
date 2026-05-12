"""CSV export for source identifier collisions across units."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "source_id",
    "unit_count",
    "unit_ids",
    "titles",
    "content_type_count",
    "first_created_date",
    "last_updated_date",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_identifier_collisions_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write rows for source identifiers mapped to multiple units."""
    unit_list = list(units)
    rows = _collision_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _collision_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[
            (
                _field_value(unit.source_project) or "Unknown",
                _field_value(unit.source_entity_type) or "Unknown",
                _field_value(unit.source_id),
            )
        ].append(unit)

    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type, source_id), group_units in sorted(
        groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]), _sort_key(item[0][2]))
    ):
        unit_ids = sorted({_field_value(unit.id) for unit in group_units if _field_value(unit.id)}, key=_sort_key)
        titles = sorted({_field_value(unit.title) for unit in group_units if _field_value(unit.title)}, key=_sort_key)
        if len(unit_ids) <= 1 and len(titles) <= 1:
            continue

        created_dates = sorted(
            value for value in (_date_value(getattr(unit, "created_at", None)) for unit in group_units) if value
        )
        updated_dates = sorted(
            value for value in (_date_value(getattr(unit, "updated_at", None)) for unit in group_units) if value
        )
        content_types = {_field_value(getattr(unit, "content_type", None)) for unit in group_units}
        content_types.discard("")
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "source_id": source_id,
                "unit_count": len(unit_ids),
                "unit_ids": "; ".join(unit_ids),
                "titles": "; ".join(titles),
                "content_type_count": len(content_types),
                "first_created_date": created_dates[0].isoformat() if created_dates else "",
                "last_updated_date": updated_dates[-1].isoformat() if updated_dates else "",
            }
        )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


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


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
