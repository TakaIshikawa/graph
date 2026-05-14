"""CSV export for metadata value spread by key."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "metadata_key",
    "unit_count",
    "value_count",
    "distinct_value_count",
    "top_value",
    "top_value_count",
    "top_value_percent",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_metadata_value_spread_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write metadata value spread statistics as deterministic CSV."""
    unit_list = list(units)
    rows = _spread_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "metadata_key_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _spread_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    unit_counts: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = {}

    for unit in units:
        metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
        for raw_key, raw_value in metadata.items():
            key = _inline_text(raw_key)
            if not key:
                continue
            values = _metadata_values(raw_value)
            if not values:
                continue
            unit_counts[key] += 1
            value_counts.setdefault(key, Counter()).update(values)

    rows: list[dict[str, str | int]] = []
    for key in sorted(value_counts, key=_sort_key):
        counts = value_counts[key]
        value_count = sum(counts.values())
        top_value, top_count = sorted(counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))[0]
        rows.append(
            {
                "metadata_key": key,
                "unit_count": unit_counts[key],
                "value_count": value_count,
                "distinct_value_count": len(counts),
                "top_value": top_value,
                "top_value_count": top_count,
                "top_value_percent": _decimal(top_count * 100 / value_count),
            }
        )
    return rows


def _metadata_values(value: object) -> list[str]:
    if isinstance(value, list | tuple | set):
        return sorted(
            {text for item in value for text in _metadata_values(item)},
            key=_sort_key,
        )
    text = _inline_text(value)
    return [text] if text else []


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
