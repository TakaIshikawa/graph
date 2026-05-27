"""Summarize declared languages on fenced code blocks in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, metadata, sort_key

_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})[ \t]*([^`\s~]*)?")


def summarize_unit_code_block_languages(units: Iterable[Any]) -> dict[str, Any]:
    """Return deterministic counts for fenced code block languages."""
    total_units = units_with_code_blocks = total_code_blocks = missing_language_count = 0
    language_counts: Counter[str] = Counter()

    for unit in units:
        total_units += 1
        blocks = _fenced_languages(_content(unit))
        if blocks:
            units_with_code_blocks += 1
        total_code_blocks += len(blocks)
        missing_language_count += sum(1 for language in blocks if not language)
        language_counts.update(language for language in blocks if language)

    return {
        "total_units": total_units,
        "units_with_code_blocks": units_with_code_blocks,
        "total_code_blocks": total_code_blocks,
        "missing_language_count": missing_language_count,
        "language_counts": [
            {"language": language, "count": language_counts[language]}
            for language in sorted(language_counts, key=sort_key)
        ],
    }


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
