"""Summarize TODO-style keywords in Markdown unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEFAULT_KEYWORDS = ("TODO", "FIXME", "HACK", "BUG", "NOTE")


def summarize_unit_markdown_todo_keywords(
    units: Iterable[Mapping[str, Any] | object], keywords: Sequence[str] | None = None, sample_limit: int = 5
) -> dict[str, Any]:
    """Summarize TODO-style keyword occurrences outside fenced code."""
    wanted = tuple(dict.fromkeys((keyword.upper() for keyword in (keywords or _DEFAULT_KEYWORDS) if keyword)))
    pattern = re.compile(r"\b(" + "|".join(re.escape(keyword) for keyword in wanted) + r")\b", re.IGNORECASE) if wanted else None
    unit_list = list(units)
    keyword_counts: Counter[str] = Counter()
    per_unit: dict[str, Counter[str]] = {}
    samples: list[dict[str, str | int]] = []
    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        counts: Counter[str] = Counter()
        if pattern:
            for line_number, line in _lines(_content(unit)):
                for match in pattern.finditer(line):
                    keyword = match.group(1).upper()
                    keyword_counts[keyword] += 1
                    counts[keyword] += 1
                    samples.append({"unit_id": uid, "line_number": line_number, "keyword": keyword, "text": line.strip()})
        if counts:
            per_unit[uid] = counts
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["keyword"])))
    return {
        "total_units": len(unit_list),
        "keyword_counts": dict(sorted(keyword_counts.items(), key=lambda item: sort_key(item[0]))),
        "units_with_keywords": len(per_unit),
        "total_keyword_occurrences": sum(keyword_counts.values()),
        "line_samples": samples[:sample_limit],
        "per_unit_top": [
            {"unit_id": uid, "keyword": keyword, "count": count}
            for uid, counts in sorted(per_unit.items(), key=lambda item: sort_key(item[0]))
            for keyword, count in sorted(counts.items(), key=lambda item: (-item[1], sort_key(item[0])))[:1]
        ],
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
