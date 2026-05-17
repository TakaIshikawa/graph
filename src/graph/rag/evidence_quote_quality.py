"""Score evidence snippets for quote usefulness in RAG answers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_TEXT_KEYS = ("quote", "snippet", "text", "content", "excerpt")
_SOURCE_KEYS = ("citation", "citations", "source", "source_url", "url", "doi", "reference")
_ID_KEYS = ("id", "evidence_id", "result_id", "source_id")


def score_evidence_quote_quality(evidence: Iterable[Any], query: str | None = None) -> list[dict[str, Any]]:
    """Return bounded quote quality scores with explainable signals."""
    query_terms = set(_tokens(query or ""))
    rows = []
    for index, item in enumerate(evidence):
        text = _text(item)
        tokens = _tokens(text)
        strengths: list[str] = []
        warnings: list[str] = []
        score = 0.0

        if 8 <= len(tokens) <= 80:
            score += 0.3
            strengths.append("useful_length")
        else:
            warnings.append("poor_length")
        if text and text[-1:] in ".!?":
            score += 0.25
            strengths.append("complete_sentence")
        else:
            warnings.append("sentence_fragment")
        if query_terms:
            overlap = sorted(query_terms.intersection(tokens))
            score += min(len(overlap) / len(query_terms), 1.0) * 0.25
            if overlap:
                strengths.append("query_overlap")
            else:
                warnings.append("no_query_overlap")
        else:
            overlap = []
        if any(_has_value(_value(item, key)) for key in _SOURCE_KEYS):
            score += 0.2
            strengths.append("citation_present")
        else:
            warnings.append("missing_citation")

        rows.append(
            {
                "evidence_id": _result_id(item, index),
                "quality_score": round(min(score, 1.0), 3),
                "strengths": strengths,
                "warnings": warnings,
                "matched_query_terms": overlap,
            }
        )
    return rows


def _payload(item: Any) -> Any:
    if isinstance(item, tuple) and item:
        return item[0]
    return item


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _value(item: Any, key: str) -> Any:
    payload = _payload(item)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key, _MISSING)
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _text(item: Any) -> str:
    if isinstance(_payload(item), str):
        return _string(_payload(item)) or ""
    for key in _TEXT_KEYS:
        text = _string(_value(item, key))
        if text is not None:
            return text
    return ""


def _tokens(value: Any) -> list[str]:
    text = _string(value)
    if text is None:
        return []
    return [token for token in TOKEN_RE.findall(text.casefold()) if len(token) > 1 and token not in COMMON_STOPWORDS]


def _has_value(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return any(_has_value(item) for item in value)
    return True


def _result_id(item: Any, index: int) -> str:
    for key in _ID_KEYS:
        text = _string(_value(item, key))
        if text is not None:
            return text
    return f"evidence-{index + 1}"
