"""CSV export helpers for tag usage by source project."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")
_TAG_FIELD = "tag"
_TOTAL_FIELD = "total"


def export_tag_source_matrix_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_count: int = 1,
) -> str | dict[str, Any]:
    """Return or write a deterministic CSV matrix of tag counts by source project."""
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")

    rows, sources = _matrix_rows(list(units), min_count=min_count)
    text = _render_csv(rows, sources)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "tags_exported": len(rows),
        "sources_exported": len(sources),
        "min_count": min_count,
        "bytes_written": output_path.stat().st_size,
    }


def _matrix_rows(
    units: list[KnowledgeUnit],
    *,
    min_count: int,
) -> tuple[list[dict[str, int | str]], list[str]]:
    tag_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sources: set[str] = set()

    for unit in sorted(units, key=_unit_sort_key):
        source = _field_value(unit.source_project)
        sources.add(source)
        for tag in _unit_tags(unit):
            tag_source_counts[tag][source] += 1

    sorted_sources = sorted(sources, key=_sort_key)
    rows: list[dict[str, int | str]] = []
    for tag, counts in sorted(tag_source_counts.items(), key=lambda item: _sort_key(item[0])):
        total = sum(counts.values())
        if total < min_count:
            continue
        row: dict[str, int | str] = {_TAG_FIELD: tag, _TOTAL_FIELD: total}
        for source in sorted_sources:
            row[source] = counts.get(source, 0)
        rows.append(row)

    return rows, sorted_sources


def _render_csv(rows: list[dict[str, int | str]], sources: list[str]) -> str:
    output = StringIO()
    fieldnames = [_TAG_FIELD, *sources, _TOTAL_FIELD]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {field: row.get(field, 0 if field != _TAG_FIELD else "") for field in fieldnames}
        )
    return output.getvalue()


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted(
        {_inline_text(tag) for tag in unit.tags if _inline_text(tag)},
        key=_sort_key,
    )


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        _field_value(unit.source_project),
        _inline_text(unit.source_id),
        _inline_text(unit.id),
    )


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
