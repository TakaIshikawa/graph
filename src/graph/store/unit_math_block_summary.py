"""Summarize math blocks and inline math in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get

_MATH_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*(math|latex|tex)\b", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_RE = re.compile(r"(?<!\\)(?<!\$)\$([^$\n]+?)(?<!\\)\$(?!\$)")


def summarize_unit_math_blocks(units: Iterable[Any]) -> dict[str, Any]:
    total_units = units_with = inline = display = fences = 0
    for unit in units:
        total_units += 1
        content = str(get(unit, "content") or "")
        i, d, f = _counts(content)
        inline += i
        display += d
        fences += f
        if i or d or f:
            units_with += 1
    return {"total_units": total_units, "units_with_math": units_with, "inline_math_count": inline, "display_math_count": display, "math_fence_count": fences}


def _counts(content: str) -> tuple[int, int, int]:
    inline = display = fences = 0
    in_fence = False
    in_math_fence = False
    in_display = False
    for line in content.splitlines():
        if in_fence:
            if _FENCE_RE.match(line):
                in_fence = False
                in_math_fence = False
            continue
        if _MATH_FENCE_RE.match(line):
            fences += 1
            in_fence = True
            in_math_fence = True
            continue
        if _FENCE_RE.match(line):
            in_fence = True
            continue
        if line.strip().startswith("$$"):
            if not in_display:
                display += 1
            in_display = not line.strip().endswith("$$") or line.strip() == "$$"
            continue
        if in_display:
            if line.strip().endswith("$$"):
                in_display = False
            continue
        inline += len(_INLINE_RE.findall(line))
    return inline, display, fences
