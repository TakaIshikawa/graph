"""Vega-Lite timeline export helpers for dated knowledge units."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, overload

from graph.export.timelinejs import _clean_text, _date_sort_value, _scalar_text, _unit_sort_key, _unit_start_date
from graph.types.models import KnowledgeUnit


@overload
def export_units_to_vegalite_timeline(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: None = None,
) -> str: ...


@overload
def export_units_to_vegalite_timeline(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path,
) -> dict[str, Any]: ...


def export_units_to_vegalite_timeline(
    units: KnowledgeUnit | Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a self-contained Vega-Lite timeline spec for dated units."""
    unit_list = [units] if isinstance(units, KnowledgeUnit) else list(units)
    values: list[dict[str, Any]] = []
    skipped_count = 0

    for unit in unit_list:
        start = _unit_start_date(unit)
        if start is None:
            skipped_count += 1
            continue
        values.append(
            {
                "id": str(unit.id or ""),
                "title": _clean_text(unit.title) or "Untitled graph unit",
                "date": _date_sort_value(start),
                "source_project": _clean_text(_scalar_text(unit.source_project)),
                "tags": ", ".join(sorted(tag for tag in (_clean_text(_scalar_text(tag)) for tag in unit.tags) if tag)),
                "_sort": [_date_sort_value(start), *_unit_sort_key(unit)],
            }
        )

    values.sort(key=lambda value: tuple(value["_sort"]))
    for value in values:
        del value["_sort"]

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Timeline of dated knowledge units.",
        "data": {"values": values},
        "mark": {"type": "tick", "tooltip": True},
        "encoding": {
            "x": {"field": "date", "type": "temporal", "title": "Date"},
            "y": {"field": "source_project", "type": "nominal", "title": "Source"},
            "tooltip": [
                {"field": "title", "type": "nominal", "title": "Title"},
                {"field": "date", "type": "temporal", "title": "Date"},
                {"field": "source_project", "type": "nominal", "title": "Source"},
                {"field": "tags", "type": "nominal", "title": "Tags"},
                {"field": "id", "type": "nominal", "title": "ID"},
            ],
        },
    }
    text = json.dumps(spec, ensure_ascii=False, sort_keys=True, indent=2)
    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(values),
        "skipped_count": skipped_count,
        "bytes_written": output_path.stat().st_size,
    }
