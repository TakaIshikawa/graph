"""Build query coverage checklists for RAG/search results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_stopwords(stopwords: Iterable[str] | None) -> set[str]:
    ignored = set(COMMON_STOPWORDS)
    for value in stopwords or ():
        ignored.update(TOKEN_RE.findall(str(value).casefold()))
    return ignored


def _query_terms(
    query: Any,
    *,
    stopwords: Iterable[str] | None,
    min_token_length: int,
) -> list[str]:
    if query is None:
        return []

    ignored = _normalize_stopwords(stopwords)
    terms: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(str(query).casefold()):
        if len(token) < min_token_length or token in ignored or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _result_value(result: Any, key: str) -> Any:
    payload = _result_payload(result)
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


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _iter_strings(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        return [
            item
            for nested_value in value.values()
            for item in _iter_strings(nested_value)
        ]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return [
            item
            for nested_value in value
            for item in _iter_strings(nested_value)
        ]
    text = _string_value(value)
    return [] if text is None else [text]


def _candidate_fields(result: Any) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    for field in ("title", "content"):
        value = _string_value(_result_value(result, field))
        if value is not None:
            fields.append((field, value))
    fields.extend(
        ("tag", value)
        for value in _iter_strings(_result_value(result, "tags"))
    )
    fields.extend(
        ("metadata", value)
        for value in _iter_strings(_result_value(result, "metadata"))
    )
    return fields


def _token_spans(text: str) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {}
    for match in TOKEN_RE.finditer(text.casefold()):
        spans.setdefault(match.group(0), []).append(match.span())
    return spans


def _snippet(text: str, start: int, end: int, snippet_chars: int) -> str:
    if len(text) <= snippet_chars:
        return text

    extra = max(snippet_chars - (end - start), 0)
    window_start = max(0, start - extra // 2)
    window_end = min(len(text), window_start + snippet_chars)
    window_start = max(0, window_end - snippet_chars)
    return text[window_start:window_end].strip()


def build_result_coverage_checklist(
    query: Any,
    results: Iterable[Any],
    *,
    stopwords: Iterable[str] | None = None,
    min_token_length: int = 3,
    snippet_chars: int = 120,
) -> dict[str, Any]:
    """Return covered and uncovered query terms for retrieved results."""
    min_token_length_value = _validate_positive_int(
        min_token_length,
        "min_token_length",
    )
    snippet_chars_value = _validate_positive_int(snippet_chars, "snippet_chars")
    terms = _query_terms(
        query,
        stopwords=stopwords,
        min_token_length=min_token_length_value,
    )

    support: dict[str, dict[str, Any]] = {
        term: {"unit_ids": set(), "snippets": {}}
        for term in terms
    }
    for index, result in enumerate(list(results)):
        unit_id = _result_id(result, index)
        for field, text in _candidate_fields(result):
            spans = _token_spans(text)
            for term in terms:
                term_spans = spans.get(term)
                if not term_spans:
                    continue
                start, end = term_spans[0]
                support[term]["unit_ids"].add(unit_id)
                support[term]["snippets"].setdefault(
                    unit_id,
                    {
                        "unit_id": unit_id,
                        "field": field,
                        "snippet": _snippet(text, start, end, snippet_chars_value),
                    },
                )

    covered: list[dict[str, Any]] = []
    uncovered: list[str] = []
    for term in terms:
        unit_ids = sorted(support[term]["unit_ids"])
        if not unit_ids:
            uncovered.append(term)
            continue
        snippets = [
            support[term]["snippets"][unit_id]
            for unit_id in unit_ids
            if unit_id in support[term]["snippets"]
        ]
        covered.append(
            {
                "term": term,
                "supporting_unit_ids": unit_ids,
                "snippets": snippets,
            }
        )

    return {
        "query_terms": terms,
        "covered": covered,
        "uncovered": uncovered,
    }
