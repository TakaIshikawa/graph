"""Summarize token overlap between unit titles and source metadata."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_SOURCE_KEYS = ("source", "source_name", "site_name", "provider", "collection")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def summarize_unit_source_title_overlap(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = 0
    buckets: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {"high_overlap": []}

    for index, unit in enumerate(units):
        total += 1
        title = _title(unit)
        source = _source(unit)
        overlap = _overlap(title, source)
        bucket = _bucket(overlap, title, source)
        buckets[bucket] += 1
        if overlap >= 0.75 and len(examples["high_overlap"]) < limit:
            examples["high_overlap"].append({"unit_id": unit_id(unit) or str(index), "title": title, "source": source, "overlap": overlap})

    rows = [{"bucket": bucket, "count": buckets[bucket]} for bucket in sorted(buckets, key=sort_key)]
    return {"total_units": total, "overlap_buckets": rows, "examples": examples}


def _title(unit: Any) -> str:
    meta = metadata(unit)
    return field_value(get(unit, "title")) or field_value(meta.get("title")) or field_value(get(unit, "name"))


def _source(unit: Any) -> str:
    meta = metadata(unit)
    for key in _SOURCE_KEYS:
        value = field_value(get(unit, key)) or field_value(meta.get(key))
        if value:
            return value
    return ""


def _overlap(title: str, source: str) -> float:
    title_tokens = set(_tokens(title))
    source_tokens = set(_tokens(source))
    if not title_tokens or not source_tokens:
        return 0.0
    return round(len(title_tokens & source_tokens) / len(title_tokens), 2)


def _bucket(overlap: float, title: str, source: str) -> str:
    if not title:
        return "missing_title"
    if not source:
        return "missing_source"
    if overlap >= 0.75:
        return "high"
    if overlap >= 0.4:
        return "medium"
    if overlap > 0:
        return "low"
    return "none"


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(value)]
