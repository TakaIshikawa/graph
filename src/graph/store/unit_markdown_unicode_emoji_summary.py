"""Summarize Unicode emoji usage in Markdown content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def summarize_unit_markdown_unicode_emoji(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = units_with = total_emoji = 0
    counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for unit in units:
        total_units += 1
        found = _emoji(str(get(unit, "content") or ""))
        if not found:
            continue
        units_with += 1
        total_emoji += len(found)
        counts.update(found)
        if len(samples) < sample_limit:
            samples.append({"unit_id": unit_id(unit), "emoji": found[:sample_limit]})
    frequency = [{"emoji": emoji, "count": counts[emoji]} for emoji in sorted(counts, key=lambda item: (-counts[item], sort_key(item)))]
    average = total_emoji / total_units if total_units else 0
    return {"total_units": total_units, "units_with_emoji": units_with, "total_emoji": total_emoji, "emoji_frequency": frequency, "emoji_per_unit_average": average, "samples": samples}


def _emoji(content: str) -> list[str]:
    found: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            found.extend(_EMOJI_RE.findall(line))
    return found
