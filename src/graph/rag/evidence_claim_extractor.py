"""Extract likely evidence claims from retrieved RAG/search results."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

_MISSING = object()
_ID_KEYS = ("id", "unit_id", "source_id")
_TITLE_KEYS = ("title", "source_title")
_TEXT_KEYS = ("content", "text", "snippet")
_CITATION_KEYS = ("citation", "citation_url", "url", "source_url", "doi")
_CAUSE_CUES = ("because", "therefore", "drives", "causes", "leads", "increases", "reduces", "improves")


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
    return _first(result, _ID_KEYS) or f"result-{index + 1}"


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) >= 20
    ]


def _signals(sentence: str, citation: str | None) -> list[str]:
    lowered = sentence.casefold()
    signals: list[str] = []
    if re.search(r"\b\d+(?:[.,]\d+)?%?\b", sentence):
        signals.append("number")
    if re.search(r"\b(?:19|20)\d{2}\b", sentence):
        signals.append("date")
    if any(cue in lowered for cue in _CAUSE_CUES):
        signals.append("causal-cue")
    if citation is not None:
        signals.append("citation")
    return signals


def _normalize(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", sentence.casefold()).strip()


def extract_evidence_claims(results: Iterable[Any], *, max_claims: int = 20) -> list[dict[str, Any]]:
    """Extract ranked claim-like sentences from retrieved content."""
    if not isinstance(max_claims, int) or isinstance(max_claims, bool) or max_claims < 0:
        raise ValueError("max_claims must be a non-negative integer")
    if max_claims == 0:
        return []

    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for result_index, result in enumerate(results):
        text = _first(result, _TEXT_KEYS)
        if text is None:
            continue
        result_id = _result_id(result, result_index)
        source_title = _first(result, _TITLE_KEYS)
        citation = _first(result, _CITATION_KEYS)
        for sentence_index, sentence in enumerate(_sentences(text)):
            normalized = _normalize(sentence)
            if not normalized or normalized in seen:
                continue
            signals = _signals(sentence, citation)
            if not signals:
                continue
            seen.add(normalized)
            score = sum({"citation": 4, "number": 3, "date": 2, "causal-cue": 2}[signal] for signal in signals)
            candidates.append(
                (
                    -score,
                    result_index,
                    sentence_index,
                    {
                        "claim": sentence,
                        "result_id": result_id,
                        "source_title": source_title,
                        "signals": signals,
                        "citation": citation,
                    },
                )
            )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [candidate[-1] for candidate in candidates[:max_claims]]
