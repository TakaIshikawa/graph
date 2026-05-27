"""Estimate reading time for unit content."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_BUCKETS = ("0 min", "<1 min", "1-2 min", "3-5 min", "6+ min")


def summarize_unit_reading_time(units: Iterable[Any], words_per_minute: int = 200) -> dict[str, Any]:
    if words_per_minute <= 0:
        raise ValueError("words_per_minute must be positive")

    word_counts = [_word_count(_content(unit)) for unit in units]
    minutes = [count / words_per_minute for count in word_counts]
    buckets = Counter(_bucket(value) for value in minutes)
    rounded_minutes = [math.ceil(value) if value else 0 for value in minutes]

    return {
        "total_units": len(word_counts),
        "total_words": sum(word_counts),
        "zero_word_units": sum(1 for count in word_counts if count == 0),
        "min_minutes": min(rounded_minutes) if rounded_minutes else 0,
        "max_minutes": max(rounded_minutes) if rounded_minutes else 0,
        "average_minutes": round((sum(minutes) / len(minutes)) if minutes else 0.0, 2),
        "bucket_distribution": [{"bucket": bucket, "count": buckets[bucket]} for bucket in _BUCKETS],
    }


def _bucket(minutes: float) -> str:
    if minutes == 0:
        return "0 min"
    if minutes < 1:
        return "<1 min"
    if minutes <= 2:
        return "1-2 min"
    if minutes <= 5:
        return "3-5 min"
    return "6+ min"


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)


def _word_count(content: str) -> int:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]+\)", r"\1", content)
    text = re.sub(r"[#*_`>|-]", " ", text)
    return len(_WORD_RE.findall(text))
