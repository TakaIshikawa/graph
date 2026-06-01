"""Summarize ATX headings that end with punctuation."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_PUNCTUATION = {":": "colon", "?": "question", "!": "exclamation", ".": "period", ",": "comma"}


def summarize_unit_markdown_heading_trailing_punctuation(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize ATX headings whose normalized text ends in selected punctuation."""
    limit = max(0, sample_limit)
    total = heading_count = punctuated = 0
    affected: set[str] = set()
    counts: Counter[str] = Counter()
    examples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, level, text, punctuation in _headings(str(get(unit, "content") or "")):
            heading_count += 1
            if not punctuation:
                continue
            punctuated += 1
            counts[punctuation] += 1
            affected.add(uid)
            examples.append({"unit_id": uid, "line": line_number, "level": level, "punctuation": punctuation, "text": text})
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), int(row["level"])))
    return {
        "total_units": total,
        "heading_count": heading_count,
        "headings_with_trailing_punctuation": punctuated,
        "affected_units": len(affected),
        "punctuation_counts": {key: counts[key] for key in sorted(counts, key=sort_key)},
        "examples": examples[:limit],
    }


def _headings(content: str) -> list[tuple[int, int, str, str]]:
    rows: list[tuple[int, int, str, str]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        match = _HEADING_RE.match(line)
        if not match:
            continue
        text = re.sub(r"\s+#+\s*$", "", match.group(2)).strip()
        punctuation = _PUNCTUATION.get(text[-1:], "")
        rows.append((line_number, len(match.group(1)), field_value(text), punctuation))
    return rows
