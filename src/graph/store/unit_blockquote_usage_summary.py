"""Summarize Markdown blockquote usage across units."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id


def summarize_unit_blockquote_usage(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_units = units_with_blockquotes = total_quote_blocks = total_quoted_lines = 0
    buckets: Counter[str] = Counter()
    top: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        total_units += 1
        lines = str(get(unit, "content") or "").splitlines()
        quoted = [line for line in lines if line.lstrip().startswith(">")]
        blocks = _blocks(lines)
        if quoted:
            units_with_blockquotes += 1
        total_quote_blocks += blocks
        total_quoted_lines += len(quoted)
        bucket = _bucket(len(quoted), len(lines))
        buckets[bucket] += 1
        top.append({"unit_id": unit_id(unit) or str(index), "title": _title(unit), "quote_blocks": blocks, "quoted_lines": len(quoted)})
    top = sorted(top, key=lambda row: (-int(row["quoted_lines"]), sort_key(row["unit_id"])))[:limit]
    return {
        "total_units": total_units,
        "units_with_blockquotes": units_with_blockquotes,
        "total_quote_blocks": total_quote_blocks,
        "total_quoted_lines": total_quoted_lines,
        "quote_density_buckets": {key: buckets[key] for key in ["none", "low", "medium", "high"] if buckets[key]},
        "top_units_by_quoted_lines": top,
    }


def _blocks(lines: list[str]) -> int:
    blocks = 0
    in_block = False
    for line in lines:
        is_quote = line.lstrip().startswith(">")
        if is_quote and not in_block:
            blocks += 1
        in_block = is_quote
    return blocks


def _bucket(quoted: int, total: int) -> str:
    if quoted == 0:
        return "none"
    density = quoted / max(total, 1)
    if density < 0.25:
        return "low"
    if density < 0.5:
        return "medium"
    return "high"


def _title(unit: Any) -> str:
    return field_value(get(unit, "title") or metadata(unit).get("title"))
