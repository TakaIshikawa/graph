"""Build deterministic query revision suggestions from RAG analysis output."""

from __future__ import annotations

from typing import Any

from graph.rag._analysis_utils import iter_strings, string

_WARNING_MAP = {
    "stale_evidence": ("refresh_time_filter", 1, "Add a recent-date filter for fresher evidence.", {"date": "recent"}),
    "stale_results": ("refresh_time_filter", 1, "Add a recent-date filter for fresher evidence.", {"date": "recent"}),
    "narrow_source_set": ("broaden_sources", 2, "Broaden retrieval across independent sources.", {"source": "independent"}),
    "too_few_sources": ("broaden_sources", 2, "Broaden retrieval across independent sources.", {"source": "independent"}),
    "weak_query_term_coverage": ("add_missing_terms", 2, "Add missing terms from weak coverage analysis.", {}),
    "weak_support": ("add_missing_terms", 2, "Add missing terms from weak coverage analysis.", {}),
    "missing_support": ("add_supporting_evidence", 1, "Search for direct supporting evidence.", {}),
    "missing_citations": ("citation_support", 1, "Search for citable sources.", {"citations": "required"}),
    "uncited_claims": ("citation_support", 1, "Search for citable sources.", {"citations": "required"}),
}


def build_query_revision_plan(query: Any, analysis: Any) -> dict[str, Any]:
    """Return prioritized revision suggestions from schema-tolerant analysis data."""
    base_query = string(query) or ""
    warnings = _warnings(analysis)
    suggestions = {}
    for warning in warnings:
        if warning not in _WARNING_MAP:
            continue
        code, priority, reason, filters = _WARNING_MAP[warning]
        revised = _revised_query(base_query, code, analysis)
        existing = suggestions.get(code)
        item = {
            "code": code,
            "priority": priority,
            "revised_query": revised,
            "reason": reason,
            "filters": filters,
        }
        if existing is None or priority < existing["priority"]:
            suggestions[code] = item

    if not suggestions and not warnings:
        suggestions["conservative_broadening"] = {
            "code": "conservative_broadening",
            "priority": 5,
            "revised_query": base_query,
            "reason": "No specific analysis warnings were provided.",
            "filters": {},
        }

    plan = sorted(suggestions.values(), key=lambda item: (item["priority"], item["code"]))
    return {"query": base_query, "suggestion_count": len(plan), "suggestions": plan, "ignored_warnings": [warning for warning in warnings if warning not in _WARNING_MAP]}


def _warnings(analysis: Any) -> list[str]:
    found: list[str] = []
    for item in iter_strings(analysis):
        normalized = item.casefold()
        if normalized in _WARNING_MAP or normalized.endswith("_warning"):
            if normalized not in found:
                found.append(normalized)
    return found


def _revised_query(query: str, code: str, analysis: Any) -> str:
    if code == "refresh_time_filter" and "after:" not in query.casefold():
        return f"{query} after:2024".strip()
    if code == "broaden_sources" and "independent" not in query.casefold():
        return f"{query} independent sources".strip()
    if code == "citation_support" and "citation" not in query.casefold():
        return f"{query} citable sources".strip()
    if code in {"add_missing_terms", "add_supporting_evidence"}:
        extras = [term for term in iter_strings(analysis) if len(term.split()) == 1 and term.casefold() not in query.casefold()]
        return " ".join([query, *extras[:3]]).strip()
    return query
