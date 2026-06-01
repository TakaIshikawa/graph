"""Summarize social-style mention handles in Markdown prose."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
_MENTION_RE = re.compile(r"(?<![\w.+-])@([A-Za-z0-9][A-Za-z0-9_.-]{0,62})(?![\w.-]*\.[A-Za-z]{2,}\b)")


def summarize_unit_markdown_mention_handles(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = mention_count = 0
    units_with: set[str] = set()
    counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_found = False
        for line_number, line in _prose_lines(str(get(unit, "content") or "")):
            for match in _MENTION_RE.finditer(line):
                handle = f"@{match.group(1).rstrip('.')}"
                counts[handle.casefold()] += 1
                mention_count += 1
                unit_found = True
                if len(samples) < limit:
                    samples.append({"unit_id": uid, "line_number": line_number, "handle": handle, "context": line.strip()})
        if unit_found:
            units_with.add(uid)
    return {
        "total_units": total,
        "mention_count": mention_count,
        "units_with_mentions": len(units_with),
        "handle_counts": {key: counts[key] for key in sorted(counts, key=sort_key)},
        "samples": samples,
    }


def _prose_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, _INLINE_CODE_RE.sub("", line)))
    return rows
