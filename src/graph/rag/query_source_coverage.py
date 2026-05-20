"""Plan source-category coverage for RAG queries."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import MISSING, domain_for, result_id, string, value

_CATEGORY_ORDER = ("official", "expert", "comparative", "timeline", "troubleshooting", "community", "publisher", "unknown")
_CUE_REQUIREMENTS = (
    ("comparison", re.compile(r"\b(compare|versus|vs|difference|better|pros and cons)\b", re.I), ("comparative", "expert")),
    ("timeline", re.compile(r"\b(timeline|history|over time|when did|chronology|before|after)\b", re.I), ("timeline", "official")),
    ("medical", re.compile(r"\b(medical|diagnosis|dose|treatment|symptom)\b", re.I), ("official", "expert")),
    ("legal", re.compile(r"\b(legal|lawsuit|contract|regulation|statute)\b", re.I), ("official", "expert")),
    ("financial", re.compile(r"\b(financial|investment|tax|loan|earnings)\b", re.I), ("official", "expert")),
    ("troubleshooting", re.compile(r"\b(troubleshoot|error|fix|debug|failure|root cause)\b", re.I), ("troubleshooting", "community")),
)


def plan_query_source_coverage(query: str, results: Iterable[Any] = (), *, required_categories: Iterable[str] | None = None) -> dict[str, Any]:
    """Return required, present, and missing source categories for a query."""
    query_text = " ".join(str(query or "").split())
    matched_cues = []
    required = []
    if required_categories is None:
        for cue, pattern, categories in _CUE_REQUIREMENTS:
            if pattern.search(query_text):
                matched_cues.append(cue)
                required.extend(categories)
        if not required:
            required = ["official"]
    else:
        required = list(required_categories)
    required = _dedupe(_normalize(required))

    rows = []
    for index, result in enumerate(results):
        category, reason = _category(result)
        rows.append({"result_id": result_id(result, index), "source_category": category, "reason": reason})
    present = sorted({row["source_category"] for row in rows if row["source_category"] != "unknown"}, key=_category_key)
    missing = [category for category in required if category not in present]
    counts = Counter(row["source_category"] for row in rows)
    warnings = []
    if not rows:
        warnings.append("no_results")
    if missing:
        warnings.append("missing_required_source_categories")
    return {
        "required_categories": required,
        "present_categories": present,
        "missing_categories": missing,
        "matched_cues": matched_cues,
        "category_counts": {category: counts.get(category, 0) for category in _CATEGORY_ORDER},
        "results": rows,
        "warnings": warnings,
    }


def _category(result: Any) -> tuple[str, str]:
    explicit = _text(_first_value(result, ("source_category", "source_type")))
    aliases = {
        "official": "official",
        "government": "official",
        "regulator": "official",
        "expert": "expert",
        "academic": "expert",
        "journal": "expert",
        "comparative": "comparative",
        "comparison": "comparative",
        "timeline": "timeline",
        "history": "timeline",
        "troubleshooting": "troubleshooting",
        "support": "troubleshooting",
        "community": "community",
        "forum": "community",
        "publisher": "publisher",
        "news": "publisher",
    }
    if explicit in aliases:
        return aliases[explicit], "explicit_metadata"
    domain = domain_for(result)
    if domain and domain.endswith(".gov"):
        return "official", "domain_heuristic"
    if domain and domain.endswith(".edu"):
        return "expert", "domain_heuristic"
    if domain and any(part in domain for part in ("stackoverflow", "github", "forum")):
        return "community", "domain_heuristic"
    if domain:
        return "publisher", "domain_present"
    return "unknown", "missing_source_category"


def _first_value(result: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        item = value(result, key)
        if item is not MISSING and item is not None and string(item) is not None:
            return item
    return MISSING


def _text(value_: Any) -> str | None:
    text = string(value_)
    return None if text is None else text.casefold().replace("-", "_").replace(" ", "_")


def _normalize(values: Iterable[str]) -> list[str]:
    return [text for value_ in values if (text := _text(value_))]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen = set()
    rows = []
    for value_ in values:
        if value_ not in seen:
            seen.add(value_)
            rows.append(value_)
    return rows


def _category_key(category: str) -> int:
    return _CATEGORY_ORDER.index(category) if category in _CATEGORY_ORDER else len(_CATEGORY_ORDER)
