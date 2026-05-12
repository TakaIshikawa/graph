"""Score whether retrieved RAG evidence can answer a query."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.evidence_freshness import score_evidence_freshness
from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE
from graph.rag.result_attribution_audit import audit_result_attribution

_MISSING = object()
_TEXT_KEYS = ("title", "content", "summary", "text", "snippet", "description")
_SOURCE_KEYS = ("source_project", "source", "source_name", "project")


def score_answerability(
    query: str,
    results: Iterable[Any],
    *,
    now: Any = None,
) -> dict[str, Any]:
    """Return a deterministic 0.0-1.0 answerability score and component diagnostics."""
    query_terms = _query_terms(query)
    result_list = list(results)
    payloads = [_payload(result) for result in result_list]

    focus_score, missing_terms = _focus_term_score(query_terms, payloads)
    diversity_score = _source_diversity_score(payloads)
    freshness_rows = score_evidence_freshness(payloads, now=now)
    freshness_score = _average([row["freshness_score"] for row in freshness_rows])
    attribution = audit_result_attribution(result_list)
    attribution_score = _attribution_score(attribution)

    components = {
        "focus_term_coverage": focus_score,
        "source_diversity": diversity_score,
        "freshness": freshness_score,
        "attribution_completeness": attribution_score,
    }
    score = round(sum(components.values()) / len(components), 3)

    return {
        "score": score,
        "components": components,
        "notes": _notes(
            result_count=len(result_list),
            query_terms=query_terms,
            missing_terms=missing_terms,
            components=components,
            freshness_rows=freshness_rows,
        ),
    }


def _query_terms(query: str) -> list[str]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    terms = {
        token
        for token in TOKEN_RE.findall(query.casefold())
        if len(token) > 1 and token not in COMMON_STOPWORDS
    }
    return sorted(terms)


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
        metadata = _field_value(unit, "metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _text_parts(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, item in value.items():
            parts.extend(_text_parts(key))
            parts.extend(_text_parts(item))
        return parts
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        parts = []
        for item in value:
            parts.extend(_text_parts(item))
        return parts
    text = _string(value)
    return [] if text is None else [text]


def _result_tokens(result: Any) -> set[str]:
    parts: list[str] = []
    for key in _TEXT_KEYS:
        for value in _candidate_values(result, key):
            parts.extend(_text_parts(value))
    for value in _candidate_values(result, "tags"):
        parts.extend(_text_parts(value))
    for value in _candidate_values(result, "metadata"):
        parts.extend(_text_parts(value))

    text = " ".join(parts).casefold()
    return {
        token
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token not in COMMON_STOPWORDS
    }


def _focus_term_score(query_terms: list[str], results: list[Any]) -> tuple[float, list[str]]:
    if not query_terms:
        return 0.0, []
    result_terms: set[str] = set()
    for result in results:
        result_terms.update(_result_tokens(result))
    matched = sorted(set(query_terms) & result_terms)
    missing = sorted(set(query_terms) - result_terms)
    return round(len(matched) / len(query_terms), 3), missing


def _first_value(result: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        for value in _candidate_values(result, key):
            if _string(value) is not None:
                return value
    return _MISSING


def _source_diversity_score(results: list[Any]) -> float:
    if not results:
        return 0.0
    sources = {
        source
        for result in results
        if (source := _string(_first_value(result, _SOURCE_KEYS))) is not None
        and source.casefold() != "unknown"
    }
    target = min(3, len(results))
    return round(min(len(sources), target) / target, 3)


def _average(values: Iterable[float]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return round(sum(value_list) / len(value_list), 3)


def _attribution_score(attribution: dict[str, Any]) -> float:
    coverage = attribution["field_coverage"]
    present = sum(row["present"] for row in coverage.values())
    total = present + sum(row["missing"] for row in coverage.values())
    if total == 0:
        return 0.0
    return round(present / total, 3)


def _notes(
    *,
    result_count: int,
    query_terms: list[str],
    missing_terms: list[str],
    components: dict[str, float],
    freshness_rows: list[dict[str, Any]],
) -> list[str]:
    notes: list[str] = []
    if result_count == 0:
        notes.append("no_results")
    if query_terms and missing_terms:
        notes.append("missing_focus_terms:" + ",".join(missing_terms[:5]))
    if components["source_diversity"] < 0.667:
        notes.append("limited_source_diversity")
    if not any(row["freshest_date"] for row in freshness_rows):
        notes.append("no_dated_evidence")
    elif components["freshness"] < 0.5:
        notes.append("stale_evidence")
    if components["attribution_completeness"] < 1.0:
        notes.append("incomplete_attribution")
    return notes
