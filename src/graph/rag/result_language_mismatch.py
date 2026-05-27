"""Analyze result language mismatches."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

LANGUAGE_KEYS = ("language", "lang", "locale", "detected_language")


def analyze_result_language_mismatch(results: Iterable[Mapping[str, Any] | object], query_language: str | None = None) -> dict[str, Any]:
    items = list(results)
    languages = [_result_language(result) for result in items]
    expected = _normalize_language(query_language) if query_language else _dominant_language(languages)
    language_counts = Counter(language for language in languages if language)
    matching = sum(1 for language in languages if language and expected and language == expected)
    missing = sum(1 for language in languages if not language)
    mismatches = [index for index, language in enumerate(languages) if language and expected and language != expected]
    return {
        "total_results": len(items),
        "query_language": expected or "",
        "matching_language_count": matching,
        "mismatched_language_count": len(mismatches),
        "missing_language_count": missing,
        "languages": [{"language": language, "count": count} for language, count in sorted(language_counts.items(), key=lambda item: (-item[1], item[0]))],
        "mismatch_examples": [_example(items[index], languages[index]) for index in mismatches[:5]],
    }


def _result_language(result: Mapping[str, Any] | object) -> str:
    metadata = _metadata(result)
    for key in LANGUAGE_KEYS:
        text = _normalize_language(_get(result, key)) or _normalize_language(metadata.get(key))
        if text:
            return text
    return ""


def _dominant_language(languages: list[str]) -> str:
    counts = Counter(language for language in languages if language)
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _example(result: Mapping[str, Any] | object, language: str) -> dict[str, Any]:
    return {"id": _text(_get(result, "id")) or _text(_get(result, "source_id")), "language": language, "title": _text(_get(result, "title"))}


def _normalize_language(value: object) -> str:
    text = _text(value).replace("_", "-").casefold()
    return text.split("-")[0] if text else ""


def _metadata(value: Mapping[str, Any] | object) -> Mapping[str, Any]:
    metadata = _get(value, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: Mapping[str, Any] | object, key: str) -> object:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _text(value: object) -> str:
    return "" if value is None else str(getattr(value, "value", value)).strip()
