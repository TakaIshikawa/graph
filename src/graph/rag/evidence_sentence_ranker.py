"""Rank sentence-sized evidence from retrieved RAG/search results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_TEXT_KEYS = ("content", "text", "snippet")
_ID_KEYS = ("id", "unit_id", "source_id")
_TERM_RE = re.compile(r"[\w-]+")
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def _payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            if (text := _string(value)):
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, _ID_KEYS) or f"result-{index + 1}"


def _content(result: Any) -> str:
    return " ".join(
        text
        for key in _TEXT_KEYS
        for value in _candidate_values(result, key)
        if (text := _string(value)) is not None
    )


def _query_terms(query: Any) -> list[str]:
    if query is None:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for term in _TERM_RE.findall(str(query).casefold()):
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _sentences(text: str) -> Iterable[tuple[int, str]]:
    for position, match in enumerate(_SENTENCE_RE.finditer(text)):
        sentence = _string(match.group(0))
        if sentence is not None:
            yield position, sentence


def _sentence_terms(sentence: str, query_terms: list[str]) -> list[str]:
    sentence_terms = set(_TERM_RE.findall(sentence.casefold()))
    return [term for term in query_terms if term in sentence_terms]


def _score(matched_terms: list[str], sentence: str) -> float:
    word_count = len(_TERM_RE.findall(sentence))
    compactness = len(matched_terms) / max(word_count, 1)
    return round(len(matched_terms) + compactness, 6)


def rank_evidence_sentences(
    results: Iterable[Any],
    query: Any,
    *,
    max_sentences: int = 8,
) -> list[dict[str, Any]]:
    """Return ranked evidence sentences that contain query terms."""
    if not isinstance(max_sentences, int) or isinstance(max_sentences, bool) or max_sentences < 0:
        raise ValueError("max_sentences must be a non-negative integer")

    query_terms = _query_terms(query)
    if not query_terms or max_sentences == 0:
        return []

    candidates: list[tuple[int, float, int, int, dict[str, Any]]] = []
    for result_index, result in enumerate(results):
        content = _content(result)
        if not content:
            continue
        result_id = _result_id(result, result_index)
        for position, sentence in _sentences(content):
            matched_terms = _sentence_terms(sentence, query_terms)
            if not matched_terms:
                continue
            score = _score(matched_terms, sentence)
            candidates.append(
                (
                    -len(matched_terms),
                    -score,
                    result_index,
                    position,
                    {
                        "result_id": result_id,
                        "sentence": sentence,
                        "matched_terms": matched_terms,
                        "score": score,
                        "position": position,
                    },
                )
            )

    candidates.sort(key=lambda item: item[:4])
    return [candidate[-1] for candidate in candidates[:max_sentences]]
