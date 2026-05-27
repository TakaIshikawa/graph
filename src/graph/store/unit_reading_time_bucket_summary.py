"""Summarize unit reading time into configurable buckets."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_DEFAULT_BUCKETS = (0, 1, 2, 5)


def summarize_unit_reading_time_buckets(
    units: Iterable[Any],
    *,
    words_per_minute: int = 200,
    minute_buckets: Sequence[int] = _DEFAULT_BUCKETS,
    sample_limit: int = 3,
) -> dict[str, Any]:
    if words_per_minute <= 0:
        raise ValueError("words_per_minute must be positive")
    buckets = _validated_buckets(minute_buckets)
    rows = []
    total_words = 0
    bucket_counts = {label: 0 for label in _bucket_labels(buckets)}
    samples: dict[str, list[dict[str, Any]]] = {label: [] for label in bucket_counts}

    for index, unit in enumerate(units):
        word_count = _word_count(_content(unit))
        total_words += word_count
        minutes = math.ceil(word_count / words_per_minute) if word_count else 0
        label = _bucket_label(minutes, buckets)
        bucket_counts[label] += 1
        sample = {"unit_id": unit_id(unit) or str(index), "word_count": word_count, "minutes": minutes}
        if len(samples[label]) < max(0, sample_limit):
            samples[label].append(sample)
        rows.append(sample)

    unit_count = len(rows)
    return {
        "unit_count": unit_count,
        "total_words": total_words,
        "average_words_per_unit": round(total_words / unit_count, 4) if unit_count else 0.0,
        "buckets": [
            {"bucket": label, "count": bucket_counts[label], "samples": sorted(samples[label], key=lambda row: sort_key(row["unit_id"]))[: max(0, sample_limit)]}
            for label in bucket_counts
        ],
    }


def _validated_buckets(values: Sequence[int]) -> tuple[int, ...]:
    buckets = tuple(values)
    if not buckets or buckets[0] != 0:
        raise ValueError("minute_buckets must start with 0")
    if any(value < 0 for value in buckets) or any(left >= right for left, right in zip(buckets, buckets[1:])):
        raise ValueError("minute_buckets must be increasing non-negative integers")
    return buckets


def _bucket_labels(buckets: tuple[int, ...]) -> list[str]:
    labels = ["0 min"]
    for start, end in zip(buckets[1:], buckets[2:]):
        labels.append(f"{start}-{end} min")
    labels.append(f"{buckets[-1] + 1}+ min")
    return labels


def _bucket_label(minutes: int, buckets: tuple[int, ...]) -> str:
    if minutes == 0:
        return "0 min"
    for start, end in zip(buckets[1:], buckets[2:]):
        if start <= minutes <= end:
            return f"{start}-{end} min"
    return f"{buckets[-1] + 1}+ min"


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)


def _word_count(content: str) -> int:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]+\)", r"\1", content)
    text = re.sub(r"[#*_`>|-]", " ", text)
    return len(_WORD_RE.findall(text))
