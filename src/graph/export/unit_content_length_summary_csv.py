"""CSV export for grouped unit content length summary statistics."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "unit_count",
    "empty_content_count",
    "min_length",
    "median_length",
    "max_length",
    "average_length",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_content_length_summary_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write grouped content length summary statistics."""
    unit_list = list(units)
    rows = _summary_rows(unit_list)
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


def _summary_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    empty_counts: Counter[tuple[str, str]] = Counter()

    for unit in units:
        group_key = (_unit_source(unit), _unit_type(unit))
        length = _content_length(unit)
        groups[group_key].append(length)
        if length == 0:
            empty_counts[group_key] += 1

    rows: list[dict[str, str | int]] = []
    for source_project, entity_type in sorted(
        groups,
        key=lambda key: (_sort_key(key[0]), _sort_key(key[1])),
    ):
        lengths = sorted(groups[(source_project, entity_type)])
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": entity_type,
                "unit_count": len(lengths),
                "empty_content_count": empty_counts[(source_project, entity_type)],
                "min_length": min(lengths),
                "median_length": _decimal(_median(lengths)),
                "max_length": max(lengths),
                "average_length": _decimal(sum(lengths) / len(lengths)),
            }
        )
    return rows


def _content_length(unit: KnowledgeUnit) -> int:
    return len(_inline_text(getattr(unit, "content", "")))


def _median(values: list[int]) -> float:
    midpoint = len(values) // 2
    if len(values) % 2:
        return float(values[midpoint])
    return (values[midpoint - 1] + values[midpoint]) / 2


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_entity_type) or "Unknown"


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
