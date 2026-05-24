"""Detect regulatory and compliance constraints in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_FRAMEWORKS: dict[str, tuple[tuple[str, re.Pattern[str]], tuple[str, ...]]] = {
    "HIPAA": ((("hipaa", re.compile(r"\bhipaa\b", re.I)), ("health privacy", re.compile(r"\bprotected health information|health privacy|phi\b", re.I))), ("regulator guidance", "official compliance text", "legal analysis")),
    "GDPR": ((("gdpr", re.compile(r"\bgdpr\b", re.I)), ("data protection", re.compile(r"\bgeneral data protection regulation|data protection authority\b", re.I))), ("official regulation", "supervisory authority guidance", "legal analysis")),
    "SEC": ((("sec", re.compile(r"\bsec\b|securities and exchange commission", re.I)), ("securities", re.compile(r"\bsecurities (?:filing|disclosure|regulation)\b", re.I))), ("regulator guidance", "primary filings", "legal analysis")),
    "OSHA": ((("osha", re.compile(r"\bosha\b", re.I)), ("workplace safety", re.compile(r"\bworkplace safety|occupational safety\b", re.I))), ("regulator guidance", "official standards", "legal analysis")),
    "FDA": ((("fda", re.compile(r"\bfda\b|food and drug administration", re.I)), ("drug approval", re.compile(r"\bdrug approval|medical device|clinical trial regulation\b", re.I))), ("regulator guidance", "official labeling", "legal analysis")),
    "SOC 2": ((("soc 2", re.compile(r"\bsoc\s*2\b|\bsoc\s+ii\b", re.I)), ("trust services", re.compile(r"\btrust services criteria|service organization controls\b", re.I))), ("audit report", "control criteria", "compliance attestation")),
}

_JURISDICTIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("United States", re.compile(r"\b(?:united states|u\.s\.|us|us federal|federal)\b", re.I)),
    ("European Union", re.compile(r"\b(?:european union|e\.u\.|eu)\b", re.I)),
    ("California", re.compile(r"\bcalifornia\b|\bccpa\b", re.I)),
    ("United Kingdom", re.compile(r"\b(?:united kingdom|u\.k\.|uk)\b", re.I)),
)


def detect_query_regulatory_constraints(query: str) -> dict[str, Any]:
    """Return matched regulatory frameworks, jurisdiction hints, and source requirements."""
    normalized = _normalize_query(query)
    framework_matches = []
    for framework, (cues, source_classes) in _FRAMEWORKS.items():
        labels = [label for label, pattern in cues if pattern.search(normalized)]
        if labels:
            framework_matches.append({"framework": framework, "matched_cues": labels, "required_source_classes": list(source_classes)})
    jurisdictions = [name for name, pattern in _JURISDICTIONS if pattern.search(normalized)]
    required = sorted({source for match in framework_matches for source in match["required_source_classes"]})
    confidence = _confidence(len(framework_matches), len(jurisdictions))
    return {
        "frameworks": framework_matches,
        "jurisdiction_hints": jurisdictions,
        "required_source_classes": required,
        "confidence": confidence,
        "rationale": _rationale(framework_matches, jurisdictions, confidence),
        "normalized_query": normalized,
    }


def _confidence(framework_count: int, jurisdiction_count: int) -> float:
    if not framework_count and not jurisdiction_count:
        return 0.0
    return round(min(0.95, 0.35 + framework_count * 0.2 + min(jurisdiction_count, 2) * 0.1), 2)


def _rationale(frameworks: list[dict[str, Any]], jurisdictions: list[str], confidence: float) -> list[str]:
    reasons = [f"matched_framework:{match['framework']}" for match in frameworks]
    reasons.extend(f"jurisdiction_hint:{jurisdiction}" for jurisdiction in jurisdictions)
    if confidence == 0.0:
        reasons.append("no_regulatory_cues")
    return reasons


def _normalize_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return " ".join(query.casefold().split())
