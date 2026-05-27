"""Detect explicit domain constraints in RAG queries."""

from __future__ import annotations

import re
from typing import Any

_DOMAIN_SPECS: tuple[tuple[str, bool, re.Pattern[str]], ...] = (
    ("healthcare", True, re.compile(r"\b(?:health(?:care)?|medical|clinical|patient|doctor|hospital|diagnos(?:e|is)|treatment|drug|medication|fda)\b", re.I)),
    ("legal", True, re.compile(r"\b(?:legal|law|lawyer|attorney|court|contract|liability|compliance|regulation|statute|gdpr|hipaa)\b", re.I)),
    ("finance", True, re.compile(r"\b(?:finance|financial|investment|investing|tax|accounting|loan|mortgage|insurance|banking|portfolio|sec)\b", re.I)),
    ("security", True, re.compile(r"\b(?:security|cybersecurity|vulnerability|exploit|malware|phishing|ransomware|encryption|authentication|zero[-\s]?day)\b", re.I)),
    ("education", False, re.compile(r"\b(?:education|educational|school|student|teacher|curriculum|lesson|classroom|university|college)\b", re.I)),
    ("scientific", False, re.compile(r"\b(?:scientific|science|research|study|experiment|peer[-\s]?reviewed|methodology|dataset|statistical|laboratory)\b", re.I)),
    ("policy", False, re.compile(r"\b(?:policy|public\s+policy|governance|government|legislation|agency|rulemaking|public\s+sector)\b", re.I)),
)


def detect_query_domain_constraint(query: str) -> dict[str, Any]:
    """Return domain labels, cue matches, and high-stakes care requirement."""
    text = _inline_text(query)
    matches = _domain_matches(text)
    domains = _unique_domains(matches)
    high_stakes = {domain for domain, requires_care, _pattern in _DOMAIN_SPECS if requires_care}
    return {
        "domains": domains,
        "matched_cues": matches,
        "requires_domain_care": any(domain in high_stakes for domain in domains),
    }


def _domain_matches(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain, requires_care, pattern in _DOMAIN_SPECS:
        for match in pattern.finditer(text):
            rows.append(
                {
                    "domain": domain,
                    "cue": match.group(0),
                    "requires_care": requires_care,
                    "span": [match.start(), match.end()],
                }
            )
    return sorted(rows, key=lambda row: (row["span"][0], row["span"][1], row["domain"]))


def _unique_domains(matches: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    domains: list[str] = []
    for match in matches:
        domain = match["domain"]
        if domain not in seen:
            seen.add(domain)
            domains.append(domain)
    return domains


def _inline_text(value: object) -> str:
    return " ".join(("" if value is None else str(value)).split())
