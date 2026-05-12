"""CSV export for confidence summaries grouped by tag."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "tag",
    "unit_count",
    "source_project_count",
    "average_confidence",
    "low_confidence_unit_count",
    "missing_confidence_unit_count",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_tag_confidence_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_units: int = 1,
) -> str | dict[str, Any]:
    """Return or write unit confidence statistics grouped by normalized tag."""
    _validate_min_units(min_units)

    unit_list = list(units)
    rows = _summary_rows(unit_list, min_units=min_units)
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
        "min_units": min_units,
        "bytes_written": output_path.stat().st_size,
    }


def _summary_rows(
    units: list[KnowledgeUnit],
    *,
    min_units: int,
) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in sorted(units, key=_unit_sort_key):
        for tag in _unit_tags(unit):
            groups[tag].append(unit)

    rows: list[dict[str, str | int]] = []
    for tag in sorted(groups, key=_sort_key):
        tagged_units = groups[tag]
        if len(tagged_units) < min_units:
            continue

        confidence_values = [_confidence_value(unit.confidence) for unit in tagged_units]
        confidences = [value for value in confidence_values if value is not None]
        rows.append(
            {
                "tag": tag,
                "unit_count": len(tagged_units),
                "source_project_count": len({_unit_source(unit) for unit in tagged_units}),
                "average_confidence": _decimal(sum(confidences) / len(confidences))
                if confidences
                else "",
                "low_confidence_unit_count": sum(1 for value in confidences if value < 0.5),
                "missing_confidence_unit_count": len(tagged_units) - len(confidences),
            }
        )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_min_units(min_units: int) -> None:
    if not isinstance(min_units, int) or isinstance(min_units, bool) or min_units < 1:
        raise ValueError("min_units must be a positive integer")


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted(
        {_inline_text(tag) for tag in unit.tags if _inline_text(tag)},
        key=_sort_key,
    )


def _confidence_value(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    return (
        _sort_key(_unit_source(unit)),
        _sort_key(unit.source_id),
        _sort_key(unit.id),
    )


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
