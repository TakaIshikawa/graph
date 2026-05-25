"""Extract success criteria from RAG queries."""

from __future__ import annotations

import re
from typing import Any

_EXPLICIT_RE = re.compile(
    r"\b(?:success\s+means|successful\s+if|acceptable\s+if|must\s+achieve|target(?:s)?|kpis?|acceptance\s+criteria)\b[^.?!;]*",
    re.I,
)
_THRESHOLD_RE = re.compile(
    r"\b(?:at\s+least|at\s+most|no\s+more\s+than|less\s+than|under|over|above|below|>=|<=|>|<)\s*\$?\d+(?:\.\d+)?\s*(?:%|percent|ms|seconds?|days?|hours?|k|m|usd|dollars?)?(?=\s|[.?!;,]|$)",
    re.I,
)
_IMPLIED_RE = re.compile(r"\b(?:evaluate|assess|is\s+it\s+good|works?|ready|viable|worth\s+it)\b", re.I)


def detect_query_success_criteria(query: str) -> dict[str, Any]:
    """Return explicit and implied success criteria from a query."""
    text = " ".join(str(query or "").split())
    criteria = _matches(text, _EXPLICIT_RE, "explicit")
    thresholds = _matches(text, _THRESHOLD_RE, "threshold")
    explicit = bool(criteria or thresholds)
    return {
        "has_explicit_success_criteria": explicit,
        "criteria": criteria,
        "numeric_thresholds": thresholds,
        "implied_criteria": bool(_IMPLIED_RE.search(text)) and not explicit,
        "warnings": [] if explicit or not _IMPLIED_RE.search(text) else ["success_criteria_implied_but_not_explicit"],
    }


def _matches(text: str, pattern: re.Pattern[str], kind: str) -> list[dict[str, Any]]:
    return [{"text": m.group(0).strip(), "type": kind, "span": [m.start(), m.end()]} for m in pattern.finditer(text)]
