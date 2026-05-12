"""CSV export for unit content length summaries."""

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
    "min_chars",
    "max_chars",
    "average_chars",
    "empty_content_count",
    "length_buckets",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_content_length_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    bucket_size: int = 1000,
) -> str | dict[str, Any]:
    """Return or write content length statistics grouped by source and entity type."""
    _validate_bucket_size(bucket_size)

    unit_list = list(units)
    rows = _summary_rows(unit_list, bucket_size=bucket_size)
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
        "bucket_size": bucket_size,
        "bytes_written": output_path.stat().st_size,
    }


def _summary_rows(
    units: list[KnowledgeUnit],
    *,
    bucket_size: int,
) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    empty_counts: Counter[tuple[str, str]] = Counter()

    for unit in sorted(units, key=_unit_sort_key):
        group_key = (_unit_source(unit), _unit_type(unit))
        length = _unit_text_length(unit)
        groups[group_key].append(length)
        if length == 0:
            empty_counts[group_key] += 1

    rows: list[dict[str, str | int]] = []
    for source_project, entity_type in sorted(
        groups,
        key=lambda key: (_sort_key(key[0]), _sort_key(key[1])),
    ):
        lengths = groups[(source_project, entity_type)]
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": entity_type,
                "unit_count": len(lengths),
                "min_chars": min(lengths),
                "max_chars": max(lengths),
                "average_chars": _decimal(sum(lengths) / len(lengths)),
                "empty_content_count": empty_counts[(source_project, entity_type)],
                "length_buckets": _bucket_text(lengths, bucket_size),
            }
        )
    return rows


def _bucket_text(lengths: list[int], bucket_size: int) -> str:
    buckets: Counter[int] = Counter(length // bucket_size for length in lengths)
    parts: list[str] = []
    for index in sorted(buckets):
        start = index * bucket_size
        end = start + bucket_size - 1
        parts.append(f"{start}-{end}:{buckets[index]}")
    return "; ".join(parts)


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_bucket_size(bucket_size: int) -> None:
    if not isinstance(bucket_size, int) or isinstance(bucket_size, bool) or bucket_size < 1:
        raise ValueError("bucket_size must be a positive integer")


def _unit_text_length(unit: KnowledgeUnit) -> int:
    parts = [_inline_text(unit.title), _inline_text(unit.content)]
    return len(" ".join(part for part in parts if part))


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_type(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.source_entity_type) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    return (
        _sort_key(_unit_source(unit)),
        _sort_key(_unit_type(unit)),
        _sort_key(unit.id or unit.source_id),
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
