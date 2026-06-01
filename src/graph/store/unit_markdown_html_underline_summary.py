"""Summarize HTML underline spans in markdown unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_UNDERLINE_RE = re.compile(r"<u\b[^>]*>(.*?)</u>", re.I)


def summarize_unit_markdown_html_underlines(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_units = units_with_underline = underline_count = 0
    text_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    for index, unit in enumerate(units):
        total_units += 1
        identifier = unit_id(unit) or str(index)
        unit_count = 0
        for line_number, text in _underlines(_content(unit)):
            unit_count += 1
            underline_count += 1
            text_counts[text] += 1
            if len(samples) < limit:
                samples.append({"unit_id": identifier, "line_number": line_number, "text": text})
        if unit_count:
            units_with_underline += 1

    samples.sort(key=lambda row: (sort_key(row["unit_id"]), row["line_number"], sort_key(row["text"])))
    most_common_text = None
    if text_counts:
        most_common_text = sorted(text_counts.items(), key=lambda item: (-item[1], sort_key(item[0])))[0][0]
    return {
        "total_units": total_units,
        "units_with_underline": units_with_underline,
        "underline_count": underline_count,
        "most_common_text": most_common_text,
        "samples": samples[:limit],
    }


def _underlines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _UNDERLINE_RE.finditer(line):
            text = " ".join(match.group(1).split())
            if text:
                rows.append((line_number, text))
    return rows


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)
