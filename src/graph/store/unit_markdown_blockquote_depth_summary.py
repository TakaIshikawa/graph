"""Summarize markdown blockquote nesting depths."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id


def summarize_unit_markdown_blockquote_depths(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_units = quote_line_count = max_depth = units_with_nested_blockquotes = 0
    depth_counts: Counter[int] = Counter()
    samples: list[dict[str, Any]] = []

    for index, unit in enumerate(units):
        total_units += 1
        identifier = unit_id(unit) or str(index)
        unit_nested = False
        for line_number, depth, text in _blockquote_lines(_content(unit)):
            quote_line_count += 1
            max_depth = max(max_depth, depth)
            depth_counts[depth] += 1
            if depth > 1:
                unit_nested = True
                if len(samples) < limit:
                    samples.append({"unit_id": identifier, "line_number": line_number, "depth": depth, "text": text})
        if unit_nested:
            units_with_nested_blockquotes += 1

    samples.sort(key=lambda row: (sort_key(row["unit_id"]), row["line_number"], row["depth"], sort_key(row["text"])))
    return {
        "total_units": total_units,
        "quote_line_count": quote_line_count,
        "max_depth": max_depth,
        "depth_counts": [{"depth": depth, "count": depth_counts[depth]} for depth in sorted(depth_counts)],
        "units_with_nested_blockquotes": units_with_nested_blockquotes,
        "samples": samples[:limit],
    }


def _blockquote_lines(content: str) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = line.lstrip()
        depth = 0
        while stripped.startswith(">"):
            depth += 1
            stripped = stripped[1:].lstrip()
        if depth:
            rows.append((line_number, depth, stripped))
    return rows


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)
