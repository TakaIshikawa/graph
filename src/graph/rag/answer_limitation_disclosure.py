"""Audit limitation disclosures in RAG answers."""

from __future__ import annotations

import re
from typing import Any

_LIMITATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("incomplete_evidence", (r"\bincomplete evidence\b", r"\blimited evidence\b", r"\bevidence is sparse\b")),
    ("unavailable_data", (r"\bdata (?:is|are) unavailable\b", r"\bno data\b", r"\bmissing data\b", r"\bnot publicly available\b")),
    ("narrow_scope", (r"\bnarrow scope\b", r"\blimited to\b", r"\bonly covers?\b", r"\bout of scope\b")),
    ("outdated_sources", (r"\boutdated sources?\b", r"\bnot up to date\b", r"\blast updated\b", r"\bstale\b")),
    ("applicability_constraints", (r"\bmay not apply\b", r"\bnot generalizable\b", r"\bapplicability\b", r"\bcontext[- ]dependent\b")),
)
_BROAD_QUERY_RE = re.compile(r"\b(?:compare|best|all|overall|recommend|should|across|global|comprehensive)\b", re.I)
_HEDGE_ONLY_RE = re.compile(r"\b(?:may|might|could|possibly|perhaps)\b", re.I)


def audit_answer_limitation_disclosure(query: str, answer: str) -> dict[str, Any]:
    """Return named limitation disclosures and broad-query disclosure risk."""
    normalized_query = " ".join(str(query or "").split())
    normalized_answer = " ".join(str(answer or "").split())
    limitation_rows = []
    for limitation_type, patterns in _LIMITATIONS:
        phrases = _phrases(normalized_answer, patterns)
        if phrases:
            limitation_rows.append({"limitation_type": limitation_type, "matched_phrases": phrases})
    broad_query = bool(_BROAD_QUERY_RE.search(normalized_query))
    hedge_only = bool(_HEDGE_ONLY_RE.search(normalized_answer)) and not limitation_rows
    score = 1.0 if limitation_rows else 0.25 if broad_query else 0.6
    return {
        "has_limitation_disclosure": bool(limitation_rows),
        "limitations": limitation_rows,
        "limitation_types": [row["limitation_type"] for row in limitation_rows],
        "broad_or_comparative_query": broad_query,
        "hedge_without_named_limitation": hedge_only,
        "disclosure_score": score,
    }


def _phrases(text: str, patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.group(0).strip() for match in re.finditer(pattern, text, re.I))
    return found
