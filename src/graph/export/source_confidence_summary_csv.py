"""CSV export for per-source confidence summaries."""

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
    "source_project",
    "unit_count",
    "confidence_count",
    "missing_confidence_count",
    "min_confidence",
    "max_confidence",
    "average_confidence",
    "low_confidence_count",
    "high_confidence_count",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_confidence_summary_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    low_threshold: float = 0.5,
    high_threshold: float = 0.8,
) -> str | dict[str, Any]:
    """Return or write confidence statistics grouped by source project."""
    _validate_thresholds(low_threshold, high_threshold)

    unit_list = list(units)
    rows = _summary_rows(unit_list, low_threshold=low_threshold, high_threshold=high_threshold)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_project_count": len(rows),
        "rows_exported": len(rows),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "bytes_written": output_path.stat().st_size,
    }


def _summary_rows(
    units: list[KnowledgeUnit],
    *,
    low_threshold: float,
    high_threshold: float,
) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[_unit_source(unit)].append(unit)

    rows: list[dict[str, str | int]] = []
    for source_project in sorted(groups, key=_sort_key):
        source_units = groups[source_project]
        values = [_confidence_value(unit.confidence) for unit in source_units]
        confidences = [value for value in values if value is not None]
        rows.append(
            {
                "source_project": source_project,
                "unit_count": len(source_units),
                "confidence_count": len(confidences),
                "missing_confidence_count": len(source_units) - len(confidences),
                "min_confidence": _decimal(min(confidences)) if confidences else "",
                "max_confidence": _decimal(max(confidences)) if confidences else "",
                "average_confidence": _decimal(sum(confidences) / len(confidences)) if confidences else "",
                "low_confidence_count": sum(1 for value in confidences if value < low_threshold),
                "high_confidence_count": sum(1 for value in confidences if value >= high_threshold),
            }
        )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_thresholds(low_threshold: float, high_threshold: float) -> None:
    if not _is_number(low_threshold):
        raise ValueError("low_threshold must be a number between 0 and 1")
    if not _is_number(high_threshold):
        raise ValueError("high_threshold must be a number between 0 and 1")
    if not 0 <= low_threshold <= 1:
        raise ValueError("low_threshold must be between 0 and 1")
    if not 0 <= high_threshold <= 1:
        raise ValueError("high_threshold must be between 0 and 1")
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be less than high_threshold")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _confidence_value(value: object) -> float | None:
    if not _is_number(value):
        return None
    return float(value)


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


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
