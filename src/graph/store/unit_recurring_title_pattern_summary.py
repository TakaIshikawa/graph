"""Recurring title pattern summary for store units."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b")
DECIMAL_RE = re.compile(r"\b\d+\.\d+\b")
INTEGER_RE = re.compile(r"\b\d+\b")


def summarize_unit_recurring_title_patterns(units: Iterable[Mapping[str, Any] | object]) -> dict[str, Any]:
    total_units = 0
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for unit in units:
        total_units += 1
        title = _text(_get(unit, "title")) or _text(_metadata(unit).get("title"))
        if not title:
            continue
        groups[(_source(unit), _pattern(title))].append(title)

    pattern_summaries: list[dict[str, Any]] = []
    for (source, pattern), titles in sorted(groups.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))):
        if len(titles) < 2:
            continue
        pattern_summaries.append(
            {
                "source": source,
                "source_project": source,
                "pattern": pattern,
                "unit_count": len(titles),
                "sample_titles": sorted(titles, key=_sort_key)[:5],
            }
        )
    return {"total_units": total_units, "recurring_pattern_count": len(pattern_summaries), "pattern_summaries": pattern_summaries}


def _pattern(title: str) -> str:
    text = UUID_RE.sub("{uuid}", title)
    text = DATE_RE.sub("{date}", text)
    text = DECIMAL_RE.sub("{number}", text)
    text = INTEGER_RE.sub("{number}", text)
    return " ".join(text.split())


def _source(unit: Mapping[str, Any] | object) -> str:
    metadata = _metadata(unit)
    return _text(_get(unit, "source_project")) or _text(_get(unit, "source")) or _text(metadata.get("source")) or "unknown"


def _metadata(value: Mapping[str, Any] | object) -> Mapping[str, Any]:
    metadata = _get(value, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
