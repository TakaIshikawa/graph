"""Summarize Markdown admonition and callout usage."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get

_ADMONITION_RE = re.compile(r"^\s{0,3}([!?]{3})\s+([A-Za-z][\w-]*)")
_CALLOUT_RE = re.compile(r"^\s{0,3}>\s*\[!([A-Za-z][\w-]*)\]")


def summarize_unit_markdown_admonitions(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = 0
    by_kind: Counter[str] = Counter()
    by_syntax: Counter[str] = Counter()
    for unit in units:
        found = False
        for line in str(get(unit, "content") or "").splitlines():
            if match := _ADMONITION_RE.match(line):
                total += 1
                found = True
                by_kind[match.group(2).casefold()] += 1
                by_syntax["admonition"] += 1
            elif match := _CALLOUT_RE.match(line):
                total += 1
                found = True
                by_kind[match.group(1).casefold()] += 1
                by_syntax["obsidian_callout"] += 1
        if found:
            units_with += 1
    return {"total_admonitions": total, "units_with_admonitions": units_with, "by_kind": dict(sorted(by_kind.items())), "by_syntax": dict(sorted(by_syntax.items()))}
