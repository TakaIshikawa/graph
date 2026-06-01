"""Detect sandbox access requirements in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_SANDBOX_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("demo_environment", (r"\bdemo\s+environment\b",)),
    ("sandbox", (r"\bsandbox\b",)),
    ("test_tenant", (r"\btest\s+tenant\b",)),
    ("trial_workspace", (r"\btrial\s+workspace\b", r"\btrial\s+tenant\b")),
)
_ENVIRONMENT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("isolated_test_data", (r"\bisolated\s+test\s+data\b",)),
    ("mock_integration", (r"\bmock\s+integration\b", r"\bmocked\s+integration\b")),
    ("non_production", (r"\bnon[-\s]?production\b", r"\bnonprod\b")),
    ("staging_account", (r"\bstaging\s+account\b", r"\bstaging\s+environment\b")),
)
_INTENT = (
    r"\baccess\b",
    r"\bprovision\b",
    r"\bcreate\b",
    r"\brequest\b",
    r"\bevaluat(?:e|ion|ing)\b",
    r"\btest(?:ing)?\b",
    r"\btrial\b",
    r"\bdemo\b",
)


def detect_query_sandbox_access_requirement(query: str) -> dict[str, Any]:
    text = _normalize_query(query)
    sandbox_terms, sandbox_phrases = _collect_matches(_SANDBOX_TERMS, text)
    environment_terms, environment_phrases = _collect_matches(_ENVIRONMENT_TERMS, text)
    has_intent = any(re.search(pattern, text, re.I) for pattern in _INTENT)
    requires_access = bool((sandbox_terms or environment_terms) and has_intent)
    return {
        "requires_sandbox_access": requires_access,
        "sandbox_terms": sandbox_terms if requires_access else [],
        "environment_terms": environment_terms if requires_access else [],
        "matched_phrases": (sandbox_phrases + environment_phrases) if requires_access else [],
        "recommendations": _recommendations(requires_access, environment_terms),
        "confidence": "high" if requires_access and environment_terms else ("medium" if requires_access else "none"),
    }


def _collect_matches(specs: tuple[tuple[str, tuple[str, ...]], ...], text: str) -> tuple[list[str], list[str]]:
    terms = []
    phrases = []
    for term, patterns in specs:
        match = _first_match(patterns, text)
        if match:
            terms.append(term)
            phrases.append(match.group(0))
    return terms, phrases


def _first_match(patterns: tuple[str, ...], text: str) -> re.Match[str] | None:
    matches = [match for pattern in patterns for match in re.finditer(pattern, text, re.I)]
    return min(matches, key=lambda match: match.start()) if matches else None


def _recommendations(requires_access: bool, environment_terms: list[str]) -> list[str]:
    if not requires_access:
        return []
    recommendations = ["confirm_non_production_scope", "define_access_duration"]
    if "isolated_test_data" in environment_terms:
        recommendations.append("verify_test_data_isolation")
    if "mock_integration" in environment_terms:
        recommendations.append("document_mock_integration_limits")
    return recommendations


def _normalize_query(query: str) -> str:
    return " ".join(str(query or "").split())
