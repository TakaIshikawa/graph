"""CSV export for per-source content word count richness."""

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
    "empty_content_count",
    "short_content_count",
    "min_words",
    "max_words",
    "average_words",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_content_richness_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    short_word_threshold: int = 25,
) -> str | dict[str, Any]:
    """Return or write per-source content word count summaries."""
    if not isinstance(short_word_threshold, int) or isinstance(short_word_threshold, bool):
        raise ValueError("short_word_threshold must be a non-negative integer")
    if short_word_threshold < 0:
        raise ValueError("short_word_threshold must be a non-negative integer")

    unit_list = list(units)
    rows = _richness_rows(unit_list, short_word_threshold=short_word_threshold)
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
        "short_word_threshold": short_word_threshold,
        "bytes_written": output_path.stat().st_size,
    }


def _richness_rows(
    units: list[KnowledgeUnit],
    *,
    short_word_threshold: int,
) -> list[dict[str, str | int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for unit in units:
        groups[_unit_source(unit)].append(_word_count(unit.content))

    rows: list[dict[str, str | int]] = []
    for source_project in sorted(groups, key=_sort_key):
        word_counts = groups[source_project]
        rows.append(
            {
                "source_project": source_project,
                "unit_count": len(word_counts),
                "empty_content_count": sum(1 for count in word_counts if count == 0),
                "short_content_count": sum(1 for count in word_counts if 0 < count < short_word_threshold),
                "min_words": min(word_counts),
                "max_words": max(word_counts),
                "average_words": _decimal(sum(word_counts) / len(word_counts)),
            }
        )
    return rows


def _word_count(value: object) -> int:
    text = _inline_text(value)
    if not text:
        return 0
    return len(text.split(" "))


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
