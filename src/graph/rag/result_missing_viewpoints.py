"""Analyze missing stakeholder or viewpoint coverage in RAG results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, value

_DEFAULTS = {
    "policy": ["government", "industry", "public", "expert"],
    "product": ["customer", "vendor", "competitor", "analyst"],
    "research": ["primary study", "review", "expert", "critic"],
}
_FALLBACK = ["supporting", "critical", "expert"]


def analyze_result_missing_viewpoints(
    query: str,
    results: Iterable[Any],
    *,
    expected_viewpoints: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return present and missing viewpoint categories."""
    expected = list(expected_viewpoints or _expected_for(query))
    present = []
    haystack = " ".join(_result_terms(results)).casefold()
    for viewpoint in expected:
        if viewpoint.casefold() in haystack:
            present.append(viewpoint)

    missing = [viewpoint for viewpoint in expected if viewpoint not in present]
    score = 1.0 if not expected else round(len(present) / len(expected), 2)
    suggestions = [f"Retrieve sources representing the {viewpoint} viewpoint." for viewpoint in missing]
    return {
        "present_viewpoints": present,
        "missing_viewpoints": missing,
        "balance_score": score,
        "retrieval_suggestions": suggestions,
    }


def _expected_for(query: str) -> list[str]:
    text = query.casefold()
    for label, viewpoints in _DEFAULTS.items():
        if label in text:
            return viewpoints
    return list(_FALLBACK)


def _result_terms(results: Iterable[Any]) -> list[str]:
    terms: list[str] = []
    for row in results:
        terms.append(content_text(row))
        terms.extend(iter_strings(value(row, "viewpoints")))
        meta = metadata(row)
        if isinstance(meta, Mapping):
            terms.extend(iter_strings(meta.get("viewpoints")))
            terms.extend(iter_strings(meta.get("source_label")))
    return terms
