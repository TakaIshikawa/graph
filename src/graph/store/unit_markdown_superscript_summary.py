"""Summarize Markdown caret superscript spans."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_SUPERSCRIPT_RE = re.compile(r"(?<![\w\\])\^([^^\s][^^\n]*?[^^\s])\^|(?<![\w\\])\^([^^\s])\^")


def summarize_unit_markdown_superscripts(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize Markdown-style superscript spans outside fenced code blocks."""
    limit = max(0, sample_limit)
    total = units_with = count = 0
    texts: Counter[str] = Counter()
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        spans = _spans(str(get(unit, "content") or ""))
        if spans:
            units_with += 1
        uid = unit_id(unit)
        for line_number, text in spans:
            count += 1
            texts[text] += 1
            samples.append({"unit_id": uid, "line_number": line_number, "text": text})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["text"])))
    return {
        "total_units": total,
        "units_with_superscript": units_with,
        "superscript_count": count,
        "most_common_text": _most_common(texts),
        "samples": samples[:limit],
    }


def _spans(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _SUPERSCRIPT_RE.finditer(line):
            text = field_value(match.group(1) or match.group(2))
            if text:
                rows.append((line_number, text))
    return rows


def _most_common(counts: Counter[str]) -> str:
    return "" if not counts else min(counts, key=lambda value: (-counts[value], sort_key(value)))
