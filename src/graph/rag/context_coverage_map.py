"""Build a deterministic query-term coverage map for RAG context."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import MISSING, field_value, rounded_ratio, string
from graph.rag.keywords import TOKEN_RE


def build_context_coverage_map(query_terms: Iterable[Any], context_items: Iterable[Any]) -> dict[str, Any]:
    """Return aggregate and per-item coverage for normalized query terms."""
    terms = _query_terms(query_terms)
    items = list(context_items or [])
    item_coverage = []
    covered = set()

    for index, item in enumerate(items):
        text = _context_text(item)
        matched_terms = [term for term in terms if _term_in_text(term, text)]
        covered.update(matched_terms)
        item_coverage.append(
            {
                "item_id": _item_id(item, index),
                "source": _source(item),
                "matched_terms": matched_terms,
                "coverage_ratio": rounded_ratio(len(matched_terms), len(terms)),
            }
        )

    covered_terms = [term for term in terms if term in covered]
    uncovered_terms = [term for term in terms if term not in covered]

    return {
        "total_terms": len(terms),
        "covered_terms": covered_terms,
        "uncovered_terms": uncovered_terms,
        "term_coverage_ratio": rounded_ratio(len(covered_terms), len(terms)),
        "item_coverage": item_coverage,
        "coverage_flags": _coverage_flags(terms, items, covered_terms, uncovered_terms),
    }


def _query_terms(query_terms: Iterable[Any]) -> list[str]:
    terms = []
    seen = set()
    for value in query_terms or ():
        normalized = _normalized_term(value)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
    return terms


def _normalized_term(value: Any) -> str | None:
    text = string(value)
    if text is None:
        return None
    normalized = " ".join(TOKEN_RE.findall(text.casefold()))
    return normalized or None


def _normalized_text(value: Any) -> str:
    text = string(value)
    if text is None:
        return ""
    return " ".join(TOKEN_RE.findall(text.casefold()))


def _value(item: Any, key: str) -> Any:
    direct = field_value(item, key)
    if direct is not MISSING and direct is not None:
        return direct
    metadata = field_value(item, "metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key, MISSING)
    return direct


def _context_text(item: Any) -> str:
    if isinstance(item, str):
        return _normalized_text(item)
    parts = []
    for key in ("text", "content", "snippet"):
        value = string(_value(item, key))
        if value is not None:
            parts.append(value)
    return _normalized_text(" ".join(parts))


def _term_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    return f" {term} " in f" {text} "


def _item_id(item: Any, index: int) -> str:
    for key in ("id", "result_id", "unit_id", "source_id"):
        value = string(_value(item, key))
        if value is not None:
            return value
    return f"context-{index + 1}"


def _source(item: Any) -> str | None:
    for key in ("source", "source_id", "source_project"):
        value = string(_value(item, key))
        if value is not None:
            return value
    return None


def _coverage_flags(
    terms: list[str],
    items: list[Any],
    covered_terms: list[str],
    uncovered_terms: list[str],
) -> list[str]:
    flags = []
    if not terms:
        flags.append("empty_query_terms")
    if not items:
        flags.append("empty_context")
    if terms and not covered_terms:
        flags.append("no_terms_covered")
    elif uncovered_terms:
        flags.append("partial_term_coverage")
    elif terms:
        flags.append("full_term_coverage")
    return flags
