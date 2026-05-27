"""Summarize citation keys in Markdown unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_BRACKET_RE = re.compile(r"\[[^\]]*@[-\w:.]+[^\]]*\]")
_KEY_RE = re.compile(r"(?<![\w.])@([A-Za-z0-9][\w:.-]*)")


def summarize_unit_citation_keys(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = 0
    counts: Counter[str] = Counter()
    for unit in units:
        found = False
        for line in _content_lines(unit):
            spans = []
            for cluster in _BRACKET_RE.finditer(line):
                spans.append(cluster.span())
                for key in _KEY_RE.findall(cluster.group(0)):
                    clean = key.rstrip(".,;:!?").casefold()
                    counts[clean] += 1
                    total += 1
                    found = True
            for match in _KEY_RE.finditer(line):
                if any(start <= match.start() < end for start, end in spans):
                    continue
                key = match.group(1).rstrip(".,;:!?").casefold()
                if not any(char.isdigit() for char in key):
                    continue
                counts[key] += 1
                total += 1
                found = True
        if found:
            units_with += 1
    most_common = [{"key": key, "count": counts[key]} for key in sorted(counts, key=lambda key: (-counts[key], sort_key(key)))]
    return {"total_citations": total, "unique_key_count": len(counts), "units_with_citations": units_with, "most_common_keys": most_common}


def _content_lines(unit: Any) -> list[str]:
    rows = []
    in_fence = False
    for line in str(get(unit, "content") or "").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append(line)
    return rows
