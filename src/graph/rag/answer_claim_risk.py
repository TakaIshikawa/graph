"""Detect high-risk and weakly cited claim patterns in drafted RAG answers."""

from __future__ import annotations

import re
from typing import Any

_CITATION_RE = re.compile(r"(\[\d+\]|\[[^\]]+\]\([^)]+\)|\([A-Za-z][^)]*,\s*\d{4}\))")
_NUMBER_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:[.,]\d+)?%?|\d+(?:[.,]\d+)?%)(?!\d)")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{4}|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b", re.I)
_ABSOLUTE_RE = re.compile(r"\b(always|never|all|none|guaranteed|proves|must|only)\b", re.I)
_HIGH_STAKES_RE = re.compile(r"\b(medical|diagnosis|dose|legal|lawsuit|contract|financial|investment|tax|loan)\b", re.I)


def analyze_answer_claim_risk(answer_text: str, citations: Any = None) -> dict[str, Any]:
    """Return deterministic claim-risk records by paragraph."""
    text = str(answer_text or "").strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    require_citations = citations is not False
    risks = []

    for index, paragraph in enumerate(paragraphs):
        cues = []
        claim_text = _CITATION_RE.sub("", paragraph)
        for label, pattern in (
            ("numeric_claim", _NUMBER_RE),
            ("date_claim", _DATE_RE),
            ("absolute_claim", _ABSOLUTE_RE),
            ("high_stakes_domain", _HIGH_STAKES_RE),
        ):
            if pattern.search(claim_text):
                cues.append(label)
        if require_citations and cues and not _CITATION_RE.search(paragraph):
            cues.append("uncited_paragraph")
        if cues:
            risks.append(
                {
                    "paragraph_index": index,
                    "risk_level": _risk_level(cues),
                    "claim_types": cues,
                    "evidence": paragraph[:160],
                }
            )

    counts = {key: sum(1 for risk in risks if key in risk["claim_types"]) for key in ("numeric_claim", "date_claim", "absolute_claim", "high_stakes_domain", "uncited_paragraph")}
    warnings = []
    if counts["high_stakes_domain"]:
        warnings.append("high_stakes_claims")
    if counts["uncited_paragraph"]:
        warnings.append("uncited_claims")
    if any(risk["risk_level"] == "high" for risk in risks):
        warnings.append("high_risk_claims")

    return {
        "paragraph_count": len(paragraphs),
        "claim_count": len(risks),
        "claim_type_counts": counts,
        "risks": risks,
        "warnings": warnings,
    }


def _risk_level(cues: list[str]) -> str:
    if "high_stakes_domain" in cues or ("uncited_paragraph" in cues and len(cues) >= 3):
        return "high"
    if "uncited_paragraph" in cues or len(cues) >= 2:
        return "medium"
    return "low"
