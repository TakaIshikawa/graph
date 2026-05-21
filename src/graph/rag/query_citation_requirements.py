"""Plan citation requirements from query intent cues."""

from __future__ import annotations

import re
from typing import Any

_INTENTS = (
    ("high_stakes", re.compile(r"\b(medical|diagnosis|dose|legal|lawsuit|contract|financial|investment|tax|loan)\b", re.I)),
    ("comparison", re.compile(r"\b(compare|versus|vs\.?|better|difference|tradeoff)\b", re.I)),
    ("timeline", re.compile(r"\b(timeline|history|when|chronology|recent|latest|over time)\b", re.I)),
    ("troubleshooting", re.compile(r"\b(error|bug|fix|troubleshoot|debug|fails?|why does)\b", re.I)),
    ("statistics", re.compile(r"\b(statistics?|numbers?|percent|rate|how many|data)\b", re.I)),
    ("definition", re.compile(r"\b(what is|define|meaning of|acronym|glossary)\b", re.I)),
    ("recommendation", re.compile(r"\b(recommend|best|choose|should i|which)\b", re.I)),
)


def plan_query_citation_requirements(query: str, *, result_count: int = 0) -> dict[str, Any]:
    """Infer transparent citation requirements for a RAG query."""
    text = str(query or "").strip()
    flags = [name for name, pattern in _INTENTS if pattern.search(text)]
    required = set()
    reasons = []
    minimum = 1
    density = "normal"
    warnings = []

    if not text:
        warnings.append("empty_query")
        minimum = 0
    elif len(text.split()) < 3:
        warnings.append("short_query")

    if "high_stakes" in flags:
        required.update({"authoritative", "recent"})
        minimum = max(minimum, 3)
        density = "high"
        reasons.append("high_stakes_requires_authoritative_recent_citations")
    if "comparison" in flags:
        required.update({"comparative", "source_per_option"})
        minimum = max(minimum, 2)
        reasons.append("comparison_requires_option_level_citations")
    if "timeline" in flags:
        required.update({"dated", "chronological"})
        minimum = max(minimum, 2)
        reasons.append("timeline_requires_dated_citations")
    if "statistics" in flags:
        required.add("data_source")
        minimum = max(minimum, 2)
        reasons.append("statistics_require_data_source_citations")
    if "definition" in flags:
        required.add("definition_source")
        reasons.append("definition_requires_source_citation")
    if "recommendation" in flags:
        required.add("rationale_source")
        minimum = max(minimum, 2)
        reasons.append("recommendation_requires_rationale_citations")
    if "troubleshooting" in flags:
        required.add("implementation_source")
        reasons.append("troubleshooting_requires_implementation_citations")

    if result_count <= 0 and text:
        warnings.append("no_results_available")
    elif result_count and result_count < minimum:
        warnings.append("result_count_below_minimum_citations")

    return {
        "intent_flags": flags,
        "required_citation_types": sorted(required),
        "minimum_citations": minimum,
        "citation_density": density,
        "reasons": reasons,
        "warnings": warnings,
    }
