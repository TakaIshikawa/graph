"""Summarize Mermaid fenced diagrams in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get

_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*mermaid\b", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def summarize_unit_mermaid_diagrams(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = 0
    counts: Counter[str] = Counter()
    for unit in units:
        found = False
        for diagram_type in _diagram_types(str(get(unit, "content") or "")):
            total += 1
            found = True
            counts[diagram_type] += 1
        if found:
            units_with += 1
    return {"total_diagrams": total, "units_with_diagrams": units_with, "diagram_type_counts": dict(sorted(counts.items()))}


def _diagram_types(content: str) -> list[str]:
    types: list[str] = []
    in_mermaid = False
    first = ""
    for line in content.splitlines():
        if in_mermaid:
            if _FENCE_RE.match(line):
                types.append(_kind(first))
                in_mermaid = False
                first = ""
            elif not first and line.strip():
                first = line.strip()
            continue
        if _OPEN_RE.match(line):
            in_mermaid = True
    if in_mermaid:
        types.append(_kind(first))
    return types


def _kind(text: str) -> str:
    return text.split()[0].casefold() if text else "unknown"
