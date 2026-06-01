"""Summarize unit content language metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_LANGUAGE_KEYS = ("language", "lang", "locale", "detected_language", "content_language")
_UNKNOWN = {"", "unknown", "und", "none", "null"}


def summarize_unit_content_languages(units: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize normalized unit language metadata values."""
    unit_list = list(units)
    counts: Counter[str] = Counter()
    missing = 0
    examples_by_language: dict[str, list[str]] = {}
    for index, unit in enumerate(unit_list):
        language = _language(unit)
        if not language:
            missing += 1
            continue
        counts[language] += 1
        examples_by_language.setdefault(language, [])
        if len(examples_by_language[language]) < sample_limit:
            examples_by_language[language].append(unit_id(unit) or str(index))
    return {
        "total_units": len(unit_list),
        "units_with_language": sum(counts.values()),
        "units_missing_language": missing,
        "language_counts": dict(sorted(counts.items())),
        "examples": [
            {"language": language, "unit_ids": sorted(ids, key=sort_key)}
            for language, ids in sorted(examples_by_language.items(), key=lambda item: sort_key(item[0]))
        ],
    }


def _language(unit: Mapping[str, Any] | object) -> str:
    data = metadata(unit)
    for key in _LANGUAGE_KEYS:
        value = field_value(get(unit, key) or data.get(key)).replace("_", "-").casefold()
        if value in _UNKNOWN:
            continue
        return value.split("-", 1)[0] if "-" in value else value
    return ""
