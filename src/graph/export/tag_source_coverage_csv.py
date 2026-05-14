"""CSV export for tag coverage across source projects."""

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
    "tag",
    "unit_count",
    "distinct_source_count",
    "source_coverage_percent",
    "top_sources",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_tag_source_coverage_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write tag-centric source coverage as deterministic CSV."""
    unit_list = list(units)
    rows = _coverage_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "tag_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _coverage_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    tag_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    all_sources: set[str] = set()

    for unit in units:
        source = _unit_source(unit)
        all_sources.add(source)
        for tag in _unit_tags(unit):
            tag_source_counts[tag][source] += 1

    source_count = len(all_sources)
    rows: list[dict[str, str | int]] = []
    for tag, counts in sorted(tag_source_counts.items(), key=lambda item: _sort_key(item[0])):
        distinct_source_count = len(counts)
        rows.append(
            {
                "tag": tag,
                "unit_count": sum(counts.values()),
                "distinct_source_count": distinct_source_count,
                "source_coverage_percent": _decimal(
                    distinct_source_count * 100 / source_count if source_count else 0
                ),
                "top_sources": _top_values(counts),
            }
        )
    return rows


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)


def _top_values(value_counts: Counter[str]) -> str:
    return "; ".join(
        f"{value} ({count})"
        for value, count in sorted(value_counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))
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
