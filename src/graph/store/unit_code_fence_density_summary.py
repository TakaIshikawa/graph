"""Summarize fenced code block density in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key

_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})[ \t]*([^`\s~]*)?")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_BUCKETS = ("none", "low", "medium", "high")


def summarize_unit_code_fence_density(units: Iterable[Any]) -> dict[str, Any]:
    total_units = total_code_blocks = 0
    bucket_counts = Counter()
    language_counts = Counter()

    for unit in units:
        total_units += 1
        content = _content(unit)
        blocks = _fenced_languages(content)
        total_code_blocks += len(blocks)
        language_counts.update(language for language in blocks if language)
        words = max(1, len(_WORD_RE.findall(content)))
        bucket_counts[_bucket((len(blocks) / words) * 1000)] += 1

    return {
        "total_units": total_units,
        "total_code_blocks": total_code_blocks,
        "density_buckets": [{"bucket": bucket, "count": bucket_counts[bucket]} for bucket in _BUCKETS],
        "language_counts": [
            {"language": language, "count": language_counts[language]}
            for language in sorted(language_counts, key=sort_key)
        ],
    }


def _bucket(density: float) -> str:
    if density == 0:
        return "none"
    if density < 5:
        return "low"
    if density < 20:
        return "medium"
    return "high"


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)


def _normalize_language(language: str) -> str:
    return language.strip().casefold()


def _fenced_languages(content: str) -> list[str]:
    languages: list[str] = []
    active_marker = ""
    for line in content.splitlines():
        match = _FENCE_RE.match(line)
        if not match:
            continue
        marker = match.group(1)[0]
        if active_marker:
            if marker == active_marker:
                active_marker = ""
            continue
        active_marker = marker
        languages.append(_normalize_language(match.group(2) or ""))
    return languages
