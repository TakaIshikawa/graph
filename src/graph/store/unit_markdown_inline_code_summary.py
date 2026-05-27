"""Summarize inline Markdown code spans outside fenced blocks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TICK_RE = re.compile(r"`+")


def summarize_unit_markdown_inline_code(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    total = units_with = total_length = 0
    delimiter_counts: Counter[int] = Counter()
    code_counts: Counter[str] = Counter()
    for unit in units:
        spans = [span for line in _content_lines(str(get(unit, "content") or "")) for span in _inline_spans(line)]
        if spans:
            units_with += 1
        for code, length in spans:
            text = field_value(code)
            total += 1; total_length += len(text); delimiter_counts[length] += 1; code_counts[text] += 1
    common = [{"code": code, "count": count} for code, count in sorted(code_counts.items(), key=lambda item: (-item[1], sort_key(item[0])))[: max(0, sample_limit)]]
    return {"total_spans": total, "units_with_inline_code": units_with, "delimiter_length_counts": dict(sorted(delimiter_counts.items())), "most_common_code_spans": common, "average_code_length": round(total_length / total, 2) if total else 0}


def _content_lines(content: str) -> list[str]:
    rows: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append(line)
    return rows


def _inline_spans(line: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    pos = 0
    while match := _TICK_RE.search(line, pos):
        delimiter = match.group(0)
        end = line.find(delimiter, match.end())
        if end == -1:
            break
        code = line[match.end() : end]
        if code:
            rows.append((code, len(delimiter)))
        pos = end + len(delimiter)
    return rows
