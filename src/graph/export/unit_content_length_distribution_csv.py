"""CSV export for unit content length buckets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["bucket", "unit_count", "min_length", "max_length", "average_length", "unit_ids"]
_DEFAULT_BUCKETS = (("empty", 0), ("short", 500), ("medium", 2000), ("long", 10000), ("very_long", None))


def export_units_to_content_length_distribution_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    buckets: Sequence[tuple[str, int | None]] | None = None,
) -> str | dict[str, Any]:
    bucket_defs = _validate_buckets(buckets or _DEFAULT_BUCKETS)
    groups = {name: [] for name, _ in bucket_defs}
    unit_list = list(units)
    for unit in unit_list:
        length = len(field_value(get(unit, "content")))
        groups[_bucket(length, bucket_defs)].append((unit_id(unit), length))
    rows = []
    for name, _limit in bucket_defs:
        values = groups[name]
        lengths = [length for _id, length in values]
        rows.append({
            "bucket": name,
            "unit_count": len(values),
            "min_length": min(lengths) if lengths else "",
            "max_length": max(lengths) if lengths else "",
            "average_length": f"{(sum(lengths) / len(lengths)):.2f}" if lengths else "",
            "unit_ids": "; ".join(sorted((_id for _id, _length in values), key=sort_key)),
        })
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _validate_buckets(buckets: Sequence[tuple[str, int | None]]) -> list[tuple[str, int | None]]:
    if not buckets or buckets[-1][1] is not None:
        raise ValueError("buckets must end with an open-ended bucket")
    previous = -1
    result = []
    for name, limit in buckets:
        if not field_value(name):
            raise ValueError("bucket names must be non-empty")
        if limit is not None and limit <= previous:
            raise ValueError("bucket thresholds must increase")
        previous = limit if limit is not None else previous
        result.append((field_value(name), limit))
    return result


def _bucket(length: int, buckets: Sequence[tuple[str, int | None]]) -> str:
    for name, limit in buckets:
        if limit is None or length <= limit:
            return name
    return buckets[-1][0]
