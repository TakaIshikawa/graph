"""CSV export for unit word count distribution buckets."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv
from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["bucket", "min_words", "max_words", "unit_count", "unit_ids"]
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_BUCKETS = [
    ("0", 0, 0),
    ("1-100", 1, 100),
    ("101-500", 101, 500),
    ("501-1000", 501, 1000),
    ("1001+", 1001, None),
]


def export_unit_word_count_distribution_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write unit counts grouped by deterministic word-count buckets."""
    unit_list = list(units)
    rows = _distribution_rows(unit_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _distribution_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "unit_ids": set()})
    for unit in units:
        bucket, _minimum, _maximum = _bucket_for_count(_word_count(unit))
        buckets[bucket]["count"] += 1
        if unit_id(unit):
            buckets[bucket]["unit_ids"].add(unit_id(unit))

    rows: list[dict[str, str | int]] = []
    for bucket, minimum, maximum in _BUCKETS:
        if bucket not in buckets:
            continue
        rows.append(
            {
                "bucket": bucket,
                "min_words": minimum,
                "max_words": "" if maximum is None else maximum,
                "unit_count": buckets[bucket]["count"],
                "unit_ids": "; ".join(sorted(buckets[bucket]["unit_ids"], key=sort_key)),
            }
        )
    return rows


def _word_count(unit: KnowledgeUnit | Mapping[str, Any]) -> int:
    for key in ("content", "text", "body", "description"):
        text = field_value(get(unit, key))
        if text:
            return len(_WORD_RE.findall(text))
    return 0


def _bucket_for_count(count: int) -> tuple[str, int, int | None]:
    for bucket, minimum, maximum in _BUCKETS:
        if count >= minimum and (maximum is None or count <= maximum):
            return bucket, minimum, maximum
    return _BUCKETS[-1]

