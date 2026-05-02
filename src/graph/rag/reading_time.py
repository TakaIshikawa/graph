"""Reading effort estimates for knowledge units."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.types.models import KnowledgeUnit

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_FENCE_RE = re.compile(r"(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$")
_FENCE_MARKER_LINE_RE = re.compile(r"(?m)^[ \t]*(?:```|~~~).*$")
_CODE_LINE_RE = re.compile(
    r"""(?x)
    ^\s*(
        (?:def|class|import|from|return|if|elif|else|for|while|try|except|with)\b
        |(?:const|let|var|function|export|interface|type)\b
        |[#/]{1,2}\s
        |[}\])];,]+$
    )
    """
)


def _validate_words_per_minute(words_per_minute: int) -> int:
    if (
        not isinstance(words_per_minute, int)
        or isinstance(words_per_minute, bool)
        or words_per_minute <= 0
    ):
        raise ValueError("words_per_minute must be a positive integer")
    return words_per_minute


def _metadata_summary(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata or {}
    if not isinstance(metadata, dict):
        return ""
    summary = metadata.get("summary")
    return summary if isinstance(summary, str) else ""


def _strip_code(text: str) -> str:
    without_fences = _FENCE_RE.sub("\n", text)
    prose_lines = []
    for line in without_fences.splitlines():
        if line.startswith(("    ", "\t")):
            continue
        if _CODE_LINE_RE.match(line):
            continue
        prose_lines.append(line)
    return "\n".join(prose_lines)


def _word_count(text: str, *, include_code: bool) -> int:
    if not include_code:
        text = _strip_code(text)
    else:
        text = _FENCE_MARKER_LINE_RE.sub("", text)
    return len(_WORD_RE.findall(text))


def _reading_minutes(word_count: int, words_per_minute: int) -> float:
    return round(word_count / words_per_minute, 2)


def _unit_text(unit: KnowledgeUnit) -> str:
    return "\n".join(
        part for part in (unit.title, unit.content, _metadata_summary(unit)) if part
    )


def estimate_reading_time(
    units: Iterable[KnowledgeUnit],
    *,
    words_per_minute: int = 220,
    include_code: bool = True,
) -> dict[str, Any]:
    """Estimate per-unit and aggregate reading effort.

    The estimate counts words from each unit's title, content, and string
    ``metadata["summary"]`` value. Input order is preserved.
    """
    wpm = _validate_words_per_minute(words_per_minute)

    unit_entries = []
    total_words = 0
    for unit in units:
        word_count = _word_count(_unit_text(unit), include_code=include_code)
        total_words += word_count
        unit_entries.append(
            {
                "unit_id": unit.id,
                "title": unit.title,
                "word_count": word_count,
                "estimated_minutes": _reading_minutes(word_count, wpm),
                "tags": list(unit.tags),
            }
        )

    return {
        "units": unit_entries,
        "totals": {
            "unit_count": len(unit_entries),
            "word_count": total_words,
            "estimated_minutes": _reading_minutes(total_words, wpm),
        },
        "settings": {
            "words_per_minute": wpm,
            "include_code": include_code,
        },
    }
