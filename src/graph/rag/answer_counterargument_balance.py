"""Audit whether answers include counterarguments when the query calls for balance."""

from __future__ import annotations

import re
from typing import Any

_BALANCE_RE = re.compile(r"\b(?:however|although|on the other hand|limitation|drawback|counterargument|opposing|critics|risk|trade[- ]off)\b", re.I)
_BALANCE_QUERY_RE = re.compile(r"\b(?:compare|versus|vs\.?|recommend|should|policy|debate|pros and cons|trade[- ]off)\b", re.I)


def audit_answer_counterargument_balance(query: str, answer: str) -> dict[str, Any]:
    """Return balance cues and whether a counterargument is missing."""
    query_text = str(query or "")
    answer_text = str(answer or "")
    requires_balance = bool(_BALANCE_QUERY_RE.search(query_text))
    cues = sorted({match.group(0).lower() for match in _BALANCE_RE.finditer(answer_text)})
    score = 1.0 if not requires_balance else min(1.0, len(cues) / 2)
    return {
        "requires_counterargument": requires_balance,
        "balance_score": round(score, 2),
        "missing_counterargument": requires_balance and not cues,
        "matched_balance_cues": cues,
    }
