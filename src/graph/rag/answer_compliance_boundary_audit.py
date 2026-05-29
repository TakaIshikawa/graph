"""Audit regulated-domain answers for compliance boundary language."""

from __future__ import annotations

import re
from typing import Any

_DOMAINS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("legal", re.compile(r"\blegal|law|contract|liability|regulation|compliance\b", re.I)),
    ("medical", re.compile(r"\bmedical|clinical|diagnos(?:is|e)|treatment|patient|doctor\b", re.I)),
    ("tax", re.compile(r"\btax|irs|deduction|filing\b", re.I)),
    ("financial", re.compile(r"\bfinancial|investment|securities|portfolio|loan|banking\b", re.I)),
    ("hr", re.compile(r"\bhr\b|human\s+resources|employee|employment|hiring|termination\b", re.I)),
    ("security", re.compile(r"\bsecurity|incident\s+response|vulnerability|breach|threat\b", re.I)),
)
_BOUNDARY_RE = re.compile(
    r"\b(?:consult\s+(?:qualified\s+)?(?:counsel|attorney|doctor|clinician|tax\s+advisor|financial\s+advisor)|not\s+(?:legal|medical|tax|financial|professional)\s+advice|verify\s+with\s+(?:policy|counsel|compliance)|jurisdiction[-\s]?specific|var(?:y|ies)\s+by\s+jurisdiction)\b",
    re.I,
)


def audit_answer_compliance_boundaries(answer: str, query: str = "") -> dict[str, Any]:
    combined = f"{query or ''} {answer or ''}"
    domains = [name for name, pattern in _DOMAINS if pattern.search(combined)]
    boundary = bool(_BOUNDARY_RE.search(str(answer or "")))
    return {
        "regulated_domain": bool(domains),
        "boundary_present": boundary,
        "domains": domains,
        "recommendation": "add_compliance_boundary_language" if domains and not boundary else "",
    }
