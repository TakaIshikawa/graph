"""Summarize explicit Markdown heading anchors."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s+\{#([A-Za-z0-9_.:-]+)\}\s*#*\s*$")


def summarize_unit_markdown_heading_anchors(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = 0
    level_counts: Counter[int] = Counter()
    anchor_counts: Counter[str] = Counter()
    for unit in units:
        found = False
        for line in str(get(unit, "content") or "").splitlines():
            if match := _HEADING_RE.match(line):
                total += 1; found = True; level_counts[len(match.group(1))] += 1; anchor_counts[match.group(3)] += 1
        if found:
            units_with += 1
    duplicate_anchor_ids = [{"anchor_id": anchor, "count": count} for anchor, count in sorted(anchor_counts.items()) if count > 1]
    return {"total_anchors": total, "units_with_anchors": units_with, "duplicate_anchor_ids": duplicate_anchor_ids, "level_counts": dict(sorted(level_counts.items())), "anchor_id_counts": dict(sorted(anchor_counts.items()))}
