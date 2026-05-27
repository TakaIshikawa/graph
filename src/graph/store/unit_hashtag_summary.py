"""Summarize inline hashtags in unit content."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_TAG_RE = re.compile(r"(?<![\w/&?=])#([A-Za-z][A-Za-z0-9_-]*(?:/[A-Za-z0-9_-]+)*)")


def summarize_unit_hashtags(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = total_occurrences = 0
    counts: Counter[str] = Counter(); depths: Counter[int] = Counter(); examples=defaultdict(list)
    for unit in units:
        total_units += 1
        for tag,line in _tags(unit):
            tag=tag.casefold(); total_occurrences += 1; counts[tag] += 1; depths[tag.count('/')+1] += 1
            if len(examples[tag]) < sample_limit: examples[tag].append({"unit_id": unit_id(unit), "line": line})
    top=sorted(counts, key=lambda tag: (-counts[tag], tag))
    return {"total_units": total_units, "total_occurrences": total_occurrences, "unique_normalized_tags": len(counts), "nested_tag_depth_distribution": [{"depth": d, "count": depths[d]} for d in sorted(depths)], "top_tags": [{"tag": t, "count": counts[t], "examples": examples[t]} for t in top[:sample_limit]]}


def _tags(unit: Any) -> list[tuple[str,int]]:
    rows=[]; in_fence=False
    for line_no,line in enumerate(str(get(unit,'content') or '').splitlines(),1):
        if _FENCE_RE.match(line): in_fence=not in_fence; continue
        if not in_fence: rows.extend((m.group(1), line_no) for m in _TAG_RE.finditer(line))
    return rows
