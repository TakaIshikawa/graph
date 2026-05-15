"""Map answer sentences to likely supporting RAG/search results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()
_STOP_WORDS = {"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to", "with"}


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


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
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            value = unit_metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).split())
    return text or None


def _first(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return None


def _result_id(result: Any, index: int) -> str:
    return _first(result, ("id", "unit_id", "source_id")) or f"result-{index + 1}"


def _tokens(text: str | None) -> set[str]:
    if text is None:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9-]*", text.casefold())
        if len(token) > 2 and token not in _STOP_WORDS
    }


def _sentences(answer: Any) -> list[str]:
    text = _string(answer) or ""
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _normalize_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value if "://" in value else f"https://{value}")
    host = (parsed.hostname or "").casefold()
    if not host:
        return None
    return f"{host}{parsed.path.rstrip('/')}"


def build_answer_source_map(answer: Any, results: Iterable[Any]) -> list[dict[str, Any]]:
    """Split answer text and map each sentence to likely supporting result IDs."""
    result_rows = []
    for index, result in enumerate(results):
        title = _first(result, ("title", "source_title"))
        url = _first(result, ("url", "source_url", "citation_url", "canonical_url"))
        content = _first(result, ("content", "text", "snippet"))
        result_rows.append(
            {
                "result_id": _result_id(result, index),
                "title": title,
                "url": _normalize_url(url),
                "tokens": _tokens(" ".join(part for part in (title, content) if part)),
            }
        )

    sentence_rows: list[dict[str, Any]] = []
    for sentence in _sentences(answer):
        sentence_tokens = _tokens(sentence)
        sentence_url_text = sentence.casefold()
        matches: list[tuple[int, int, str, list[str]]] = []
        for row in result_rows:
            score = 0
            matched_terms = sorted(sentence_tokens & row["tokens"])
            if row["url"] and row["url"] in sentence_url_text:
                score += 100
            if row["title"] and row["title"].casefold() in sentence.casefold():
                score += 80
            if matched_terms:
                score += len(matched_terms)
            if score:
                matches.append((-score, result_rows.index(row), row["result_id"], matched_terms))
        matches.sort()
        support_ids = [item[2] for item in matches[:3]]
        matched = sorted({term for item in matches[:3] for term in item[3]})
        best_score = -matches[0][0] if matches else 0
        confidence = "high" if best_score >= 80 else "medium" if best_score >= 3 else "low" if best_score else "none"
        sentence_rows.append(
            {
                "sentence": sentence,
                "supporting_result_ids": support_ids,
                "matched_terms": matched,
                "confidence": confidence,
            }
        )
    return sentence_rows
