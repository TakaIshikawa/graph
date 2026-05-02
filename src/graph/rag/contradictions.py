"""Lightweight contradiction cue detection for RAG/search results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_MISSING = object()
_SNIPPET_LENGTH = 160


@dataclass(frozen=True)
class _Cue:
    label: str
    strength: float
    pattern: re.Pattern[str]


def _cue_pattern(*terms: str) -> re.Pattern[str]:
    alternatives = [re.escape(term).replace(r"\ ", r"\s+") for term in terms]
    return re.compile(
        rf"(?<![A-Za-z0-9])(?:{'|'.join(alternatives)})(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


_CUES = (
    _Cue("retracted", 5.0, _cue_pattern("retracted", "retraction")),
    _Cue("withdrawn", 4.8, _cue_pattern("withdrawn", "withdrawal")),
    _Cue("fails to replicate", 4.6, _cue_pattern("fails to replicate", "failed to replicate")),
    _Cue(
        "contradicts",
        4.2,
        _cue_pattern("contradict", "contradicts", "contradicted", "contradictory"),
    ),
    _Cue("correction", 3.6, _cue_pattern("correction", "corrections", "corrected", "corrigendum")),
    _Cue(
        "challenge",
        3.2,
        _cue_pattern(
            "challenge",
            "challenges",
            "challenged",
            "challenging",
            "dispute",
            "disputes",
            "disputed",
            "contest",
            "contests",
            "contested",
        ),
    ),
    _Cue("however", 1.4, _cue_pattern("however")),
    _Cue("but", 1.0, _cue_pattern("but")),
)


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    return limit


def _unit_value(unit: Any, key: str) -> Any:
    if isinstance(unit, Mapping):
        return unit.get(key, _MISSING)
    return getattr(unit, key, _MISSING)


def _result_value(result: Any, key: str) -> Any:
    value = _unit_value(result, key)
    if value is not _MISSING and value is not None:
        return value

    unit = _unit_value(result, "unit")
    if unit is _MISSING or unit is None:
        return value
    nested_value = _unit_value(unit, key)
    if nested_value is not _MISSING:
        return nested_value
    return value


def _text_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _bounded_snippet(text: str, start: int, end: int) -> str:
    if len(text) <= _SNIPPET_LENGTH:
        return text

    match_length = end - start
    remaining = max(_SNIPPET_LENGTH - match_length, 0)
    prefix = remaining // 2
    snippet_start = max(start - prefix, 0)
    snippet_end = min(snippet_start + _SNIPPET_LENGTH, len(text))
    snippet_start = max(snippet_end - _SNIPPET_LENGTH, 0)

    snippet = text[snippet_start:snippet_end].strip()
    if snippet_start > 0:
        snippet = f"...{snippet[3:]}" if len(snippet) > 3 else "..."
    if snippet_end < len(text):
        snippet = f"{snippet[:-3]}..." if len(snippet) > 3 else "..."
    return snippet


def _result_key(result: Any, key: str) -> str:
    value = _text_value(_result_value(result, key))
    return value or ""


def detect_contradiction_cues(
    results: Iterable[Any],
    *,
    text_keys: Iterable[str] = ("title", "content", "snippet"),
    limit: int | None = 20,
) -> list[dict[str, Any]]:
    """Return ranked local contradiction cue records for matching results.

    The detector uses deterministic keyword cues only; it does not attempt to
    infer semantic contradiction.
    """
    limit_value = _validate_limit(limit)
    text_key_list = [str(key) for key in text_keys]
    records: list[dict[str, Any]] = []

    for result in results:
        occurrences: list[dict[str, Any]] = []
        for key_index, key in enumerate(text_key_list):
            text = _text_value(_result_value(result, key))
            if text is None:
                continue
            for cue in _CUES:
                for match in cue.pattern.finditer(text):
                    occurrences.append(
                        {
                            "cue": cue.label,
                            "strength": cue.strength,
                            "text_key": key,
                            "key_index": key_index,
                            "start": match.start(),
                            "end": match.end(),
                            "text": text,
                        }
                    )

        if not occurrences:
            continue

        occurrences.sort(
            key=lambda item: (
                -item["strength"],
                item["key_index"],
                item["start"],
                item["cue"],
            )
        )
        best = occurrences[0]
        score = round(sum(item["strength"] for item in occurrences), 6)
        records.append(
            {
                "unit_id": _result_key(result, "id") or _result_key(result, "unit_id"),
                "title": _result_key(result, "title"),
                "cue": best["cue"],
                "text_key": best["text_key"],
                "snippet": _bounded_snippet(best["text"], best["start"], best["end"]),
                "score": score,
            }
        )

    records.sort(
        key=lambda item: (
            -item["score"],
            item["unit_id"],
            item["title"],
            item["cue"],
            item["text_key"],
        )
    )
    if limit_value is not None:
        return records[:limit_value]
    return records
