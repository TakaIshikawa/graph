"""Generate deterministic follow-up questions for RAG answer gaps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_TEXT_KEYS = ("title", "content", "text", "summary")
_DATE_KEYS = ("date", "created_at", "updated_at", "published_at", "publication_date", "timestamp")


def _validate_max_questions(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("max_questions must be a non-negative integer")
    return value


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _value(result: Any, key: str) -> Any:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            return unit_metadata.get(key, _MISSING)
    return value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _tokens(text: Any) -> set[str]:
    value = _string(text)
    if value is None:
        return set()
    return {
        token
        for token in TOKEN_RE.findall(value.casefold())
        if len(token) > 2 and token not in COMMON_STOPWORDS
    }


def _result_tokens(result: Any) -> set[str]:
    terms: set[str] = set()
    for key in _TEXT_KEYS:
        terms.update(_tokens(_value(result, key)))
    return terms


def _source(result: Any) -> str:
    return _string(_value(result, "source_project")) or "unknown"


def _has_date(result: Any) -> bool:
    return any(_string(_value(result, key)) is not None for key in _DATE_KEYS)


def _add(rows: list[dict[str, Any]], seen: set[str], question: str, reason: str, severity: str, evidence_count: int) -> None:
    if question in seen:
        return
    seen.add(question)
    rows.append(
        {
            "question": question,
            "reason": reason,
            "severity": severity,
            "evidence_count": evidence_count,
        }
    )


def build_answer_gap_questions(
    query: Any,
    results: Iterable[Any],
    *,
    max_questions: int = 5,
) -> list[dict[str, Any]]:
    """Generate follow-up questions for common answerability gaps."""
    max_value = _validate_max_questions(max_questions)
    result_list = list(results)
    query_terms = _tokens(query)
    covered_terms: set[str] = set()
    for result in result_list:
        covered_terms.update(_result_tokens(result))
    missing_terms = sorted(query_terms - covered_terms)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for term in missing_terms:
        _add(
            rows,
            seen,
            f"Which evidence specifically addresses {term}?",
            "missing_query_term",
            "high",
            0,
        )
    if result_list and len({_source(result) for result in result_list}) < 2:
        _add(
            rows,
            seen,
            "Can another source confirm or challenge this answer?",
            "single_source_result_set",
            "medium",
            len(result_list),
        )
    if result_list and not any(_has_date(result) for result in result_list):
        _add(rows, seen, "What dates or time ranges bound the evidence?", "missing_dates", "medium", len(result_list))
    if len(result_list) < 3:
        _add(rows, seen, "Is there enough evidence to answer confidently?", "low_evidence_count", "high", len(result_list))

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda item: (severity_rank[item["severity"]], item["reason"], item["question"]))
    return rows[:max_value]
