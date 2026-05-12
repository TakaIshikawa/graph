"""Build compact source conflict briefs for retrieved RAG results."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()
_SOURCE_PROJECT_KEYS = ("source_project", "source", "project")
_ID_KEYS = ("id", "unit_id", "source_id")
_TEXT_KEYS = ("claim", "claim_text", "snippet", "content", "text", "title")
_KEYWORD_KEYS = ("conflict_terms", "keywords", "keyphrases", "tags")
_KEYWORD_VALUE_KEYS = ("term", "keyword", "phrase", "key", "value", "tag")
_SNIPPET_LENGTH = 180

_DISAGREEMENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("contradicts", re.compile(r"\bcontradict(?:s|ed|ory)?\b", re.IGNORECASE)),
    ("disputes", re.compile(r"\b(?:dispute|disputes|disputed)\b", re.IGNORECASE)),
    ("challenges", re.compile(r"\b(?:challenge|challenges|challenged)\b", re.IGNORECASE)),
    ("retracted", re.compile(r"\b(?:retracted|retraction)\b", re.IGNORECASE)),
    ("fails to replicate", re.compile(r"\bfail(?:s|ed)? to replicate\b", re.IGNORECASE)),
    ("however", re.compile(r"\bhowever\b", re.IGNORECASE)),
    ("but", re.compile(r"\bbut\b", re.IGNORECASE)),
    ("not", re.compile(r"\b(?:not|no evidence|unsupported|inconsistent)\b", re.IGNORECASE)),
)


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_non_negative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


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
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING:
            yield metadata_value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING:
            yield unit_value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            unit_metadata_value = unit_metadata.get(key, _MISSING)
            if unit_metadata_value is not _MISSING:
                yield unit_metadata_value


def _first_value(result: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        for value in _candidate_values(result, key):
            if value is not None and not (isinstance(value, str) and not value.strip()):
                return value
    return _MISSING


def _string_value(value: Any, default: str | None = None) -> str | None:
    if value is _MISSING or value is None:
        return default
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or default


def _iter_string_values(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        for key in _KEYWORD_VALUE_KEYS:
            string = _string_value(value.get(key, _MISSING))
            if string is not None:
                return [string]
        return [_string_value(key) for key in value if _string_value(key) is not None]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        strings: set[str] = set()
        for item in value:
            strings.update(_iter_string_values(item) if isinstance(item, Mapping) else [])
            if not isinstance(item, Mapping):
                string = _string_value(item)
                if string is not None:
                    strings.add(string)
        return sorted(strings)
    string = _string_value(value)
    return [] if string is None else [string]


def _normalize_term(value: Any) -> str | None:
    text = _string_value(value)
    if text is None:
        return None
    tokens = [
        token
        for token in TOKEN_RE.findall(text.casefold())
        if token not in COMMON_STOPWORDS
    ]
    return " ".join(tokens) or None


def _explicit_terms(result: Any) -> set[str]:
    terms: set[str] = set()
    for key in _KEYWORD_KEYS:
        for value in _candidate_values(result, key):
            for item in _iter_string_values(value):
                normalized = _normalize_term(item)
                if normalized is not None:
                    terms.add(normalized)
    return terms


def _fallback_terms(result: Any, *, min_term_length: int) -> set[str]:
    text = " ".join(
        value
        for key in ("title", "content", "snippet", "claim", "claim_text")
        if (value := _string_value(_first_value(result, (key,)))) is not None
    )
    return {
        token
        for token in TOKEN_RE.findall(text.casefold())
        if len(token) >= min_term_length and token not in COMMON_STOPWORDS
    }


def _terms(result: Any, *, min_term_length: int) -> set[str]:
    explicit = _explicit_terms(result)
    if explicit:
        return explicit
    return _fallback_terms(result, min_term_length=min_term_length)


def _unit_id(result: Any, index: int) -> str:
    return _string_value(_first_value(result, _ID_KEYS), f"result-{index + 1}") or f"result-{index + 1}"


def _source_project(result: Any) -> str:
    return _string_value(_first_value(result, _SOURCE_PROJECT_KEYS), "unknown") or "unknown"


def _claim_text(result: Any) -> str:
    for key in _TEXT_KEYS:
        value = _string_value(_first_value(result, (key,)))
        if value is not None:
            return value
    return ""


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _SNIPPET_LENGTH:
        return text
    return f"{text[: _SNIPPET_LENGTH - 3].rstrip()}..."


def _disagreement_cues(text: str) -> list[str]:
    return [
        label
        for label, pattern in _DISAGREEMENT_PATTERNS
        if pattern.search(text)
    ]


def build_source_conflict_brief(
    results: Iterable[Any],
    *,
    min_source_count: int = 2,
    limit: int | None = None,
    min_term_length: int = 4,
) -> list[dict[str, Any]]:
    """Return deterministic candidate conflict rows grouped by normalized terms.

    Inputs may be ``KnowledgeUnit`` objects, dictionaries, model-like wrappers,
    or ``(unit, score)`` tuples. The helper is cue-based and surfaces evidence
    for review; it does not infer semantic contradiction.
    """
    min_sources = _validate_positive_int(min_source_count, "min_source_count")
    limit_value = _validate_non_negative_int(limit, "limit")
    min_length = _validate_positive_int(min_term_length, "min_term_length")

    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "source_projects": set(),
            "supporting_unit_ids": set(),
            "claim_snippets": {},
            "disagreement_cues": set(),
        }
    )

    for index, result in enumerate(results):
        unit_id = _unit_id(result, index)
        source_project = _source_project(result)
        claim_text = _claim_text(result)
        cues = _disagreement_cues(claim_text)
        for term in _terms(result, min_term_length=min_length):
            bucket = buckets[term]
            bucket["source_projects"].add(source_project)
            bucket["supporting_unit_ids"].add(unit_id)
            bucket["claim_snippets"].setdefault(unit_id, _snippet(claim_text))
            bucket["disagreement_cues"].update(cues)

    rows: list[dict[str, Any]] = []
    for term, bucket in buckets.items():
        source_projects = sorted(bucket["source_projects"])
        if len(source_projects) < min_sources:
            continue
        unit_ids = sorted(bucket["supporting_unit_ids"])
        cues = sorted(bucket["disagreement_cues"])
        rows.append(
            {
                "term": term,
                "source_projects": source_projects,
                "source_project_count": len(source_projects),
                "supporting_unit_ids": unit_ids,
                "unit_count": len(unit_ids),
                "claim_snippets": [
                    {"unit_id": unit_id, "snippet": bucket["claim_snippets"][unit_id]}
                    for unit_id in unit_ids
                ],
                "disagreement_cues": cues,
                "has_disagreement_cue": bool(cues),
                "confidence": "high" if cues and len(source_projects) > 1 else "review",
            }
        )

    rows.sort(
        key=lambda row: (
            0 if row["has_disagreement_cue"] else 1,
            -row["source_project_count"],
            -row["unit_count"],
            row["term"],
        )
    )
    if limit_value is not None:
        rows = rows[:limit_value]
    return rows
