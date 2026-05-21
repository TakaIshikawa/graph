"""Analyze whether retrieved results define terms requested by a query."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_QUERY_PATTERNS = (
    re.compile(r"\bwhat\s+is\s+([A-Za-z][A-Za-z0-9 -]{1,60})", re.I),
    re.compile(r"\bdefine\s+([A-Za-z][A-Za-z0-9 -]{1,60})", re.I),
    re.compile(r"\bmeaning\s+of\s+([A-Za-z][A-Za-z0-9 -]{1,60})", re.I),
    re.compile(r"\b([A-Z]{2,12})\s+acronym\b"),
    re.compile(r"\bglossary\s+for\s+([A-Za-z][A-Za-z0-9 -]{1,60})", re.I),
)


def analyze_result_definition_coverage(query: str, results: Iterable[Any]) -> dict[str, Any]:
    """Return requested query terms that are and are not defined in result text."""
    terms = _query_terms(str(query or ""))
    rows = []
    defined: set[str] = set()
    for index, result in enumerate(results or []):
        text = content_text(result)
        result_terms = [term for term in terms if _defines(text, term)]
        defined.update(result_terms)
        rows.append(
            {
                "result_id": result_id(result, index),
                "defined_terms": result_terms,
                "reasons": [] if result_terms else ["no_requested_definition"],
            }
        )

    missing = [term for term in terms if term not in defined]
    reasons = Counter()
    if not terms:
        reasons["no_definition_terms_requested"] += 1
    for _term in missing:
        reasons["missing_requested_definition"] += 1
    for row in rows:
        for reason in row["reasons"]:
            reasons[reason] += 1

    warnings = []
    if not terms:
        warnings.append("no_definition_terms_requested")
    if not rows:
        warnings.append("no_results")
    if missing:
        warnings.append("missing_requested_definitions")

    return {
        "query_terms": terms,
        "defined_terms": sorted(defined, key=str.casefold),
        "missing_terms": missing,
        "definition_count": sum(len(row["defined_terms"]) for row in rows),
        "results": rows,
        "reason_counts": dict(sorted(reasons.items())),
        "warnings": warnings,
    }


def _query_terms(query: str) -> list[str]:
    terms = []
    for pattern in _QUERY_PATTERNS:
        for match in pattern.finditer(query):
            term = re.split(r"[?.!,;:]", match.group(1).strip())[0].strip(" -")
            if term and term.casefold() not in {item.casefold() for item in terms}:
                terms.append(term)
    return terms


def _defines(text: str, term: str) -> bool:
    escaped = re.escape(term)
    pattern = re.compile(
        rf"\b{escaped}\b\s+(?:is|are|means|refers to|stands for|describes)\b|"
        rf"\b(?:is|are|means|refers to|stands for|describes)\s+\b{escaped}\b",
        re.I,
    )
    return bool(pattern.search(text))
