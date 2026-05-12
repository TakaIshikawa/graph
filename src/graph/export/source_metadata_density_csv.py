"""CSV export for per-source metadata density."""

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
    "unit_count",
    "units_with_metadata",
    "metadata_coverage_percent",
    "distinct_metadata_keys",
    "average_keys_per_unit",
    "top_metadata_keys",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_metadata_density_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_units: int = 1,
) -> str | dict[str, Any]:
    """Return or write a deterministic per-source metadata density CSV."""
    if not isinstance(min_units, int) or isinstance(min_units, bool) or min_units < 1:
        raise ValueError("min_units must be a positive integer")

    unit_list = list(units)
    rows = _density_rows(unit_list, min_units=min_units)
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
        "min_units": min_units,
        "bytes_written": output_path.stat().st_size,
    }


def _density_rows(units: list[KnowledgeUnit], *, min_units: int) -> list[dict[str, str | int]]:
    groups: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        groups[_unit_source(unit)].append(unit)

    rows: list[dict[str, str | int]] = []
    for source_project in sorted(groups, key=_sort_key):
        source_units = groups[source_project]
        if len(source_units) < min_units:
            continue

        key_counts: Counter[str] = Counter()
        total_keys = 0
        units_with_metadata = 0
        for unit in source_units:
            keys = [_inline_text(key) for key in (unit.metadata or {}) if _inline_text(key)]
            if keys:
                units_with_metadata += 1
            total_keys += len(keys)
            key_counts.update(keys)

        unit_count = len(source_units)
        rows.append(
            {
                "source_project": source_project,
                "unit_count": unit_count,
                "units_with_metadata": units_with_metadata,
                "metadata_coverage_percent": _decimal(units_with_metadata * 100 / unit_count),
                "distinct_metadata_keys": len(key_counts),
                "average_keys_per_unit": _decimal(total_keys / unit_count),
                "top_metadata_keys": _top_keys(key_counts),
            }
        )
    return rows


def _top_keys(key_counts: Counter[str]) -> str:
    return "; ".join(
        f"{key} ({count})"
        for key, count in sorted(key_counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))
    )


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


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
