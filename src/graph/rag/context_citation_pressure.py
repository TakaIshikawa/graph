"""Estimate citation pressure created by retrieved context."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()
_DATE_RE = re.compile(r"\b(?:19\d{2}|20\d{2})(?:-\d{2}-\d{2})?\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]+(?:\s+[A-Z][A-Za-z0-9&.-]+){0,2}\b")


def _payload(result: Any) -> Any:
    return result[0] if isinstance(result, tuple) and result else result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str):
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


def _content(result: Any) -> str:
    return " ".join(
        text
        for key in ("content", "text", "snippet")
        for value in _candidate_values(result, key)
        if (text := _string(value)) is not None
    )


def _source(result: Any) -> str:
    raw = _first(result, ("source", "source_project", "domain", "url", "canonical_url")) or "unknown"
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    return (parsed.hostname or raw).casefold().removeprefix("www.")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if len(part.strip()) >= 20]


def estimate_context_citation_pressure(results: Iterable[Any]) -> dict[str, Any]:
    """Return a citation-pressure score and recommendation for context."""
    result_list = list(results)
    texts = [_content(result) for result in result_list]
    sentences = [sentence for text in texts for sentence in _sentences(text)]
    factual = [
        sentence
        for sentence in sentences
        if _NUMBER_RE.search(sentence) or _DATE_RE.search(sentence) or len(_ENTITY_RE.findall(sentence)) >= 1
    ]
    numeric_count = sum(len(_NUMBER_RE.findall(text)) for text in texts)
    date_count = sum(len(_DATE_RE.findall(text)) for text in texts)
    entity_count = len({match.casefold() for text in texts for match in _ENTITY_RE.findall(text)})
    sources = {_source(result) for result in result_list if _source(result) != "unknown"}
    score = min(100, len(factual) * 8 + numeric_count * 4 + date_count * 5 + entity_count * 3 + len(sources) * 6)
    if score >= 60:
        label = "high"
    elif score >= 25:
        label = "medium"
    else:
        label = "low"
    recommended = max(1 if factual else 0, min(8, (len(factual) + 2) // 3 + (1 if len(sources) > 1 else 0)))
    return {
        "pressure_score": score,
        "label": label,
        "recommended_min_citations": recommended,
        "contributing_factors": {
            "factual_sentence_count": len(factual),
            "numeric_count": numeric_count,
            "date_count": date_count,
            "entity_count": entity_count,
            "source_count": len(sources),
        },
        "summary": {"result_count": len(result_list), "sentence_count": len(sentences)},
    }
