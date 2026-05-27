"""Classify citation recency mix and answer acknowledgement."""

from __future__ import annotations

import re
from typing import Any

from graph.rag._analysis_utils import coerce_now, result_date

_ACK_RE = re.compile(r"\b(?:older sources?|recent sources?|mixed recency|date range|stale|current as of|as of)\b", re.I)


def audit_answer_source_recency_mix(answer: str, citations: list[dict], *, now: Any = None) -> dict[str, Any]:
    today = coerce_now(now)
    dates = [day for citation in citations or [] if (day := result_date(citation)) is not None]
    if not dates:
        mix = "unknown"
    else:
        current = sum(1 for day in dates if (today - day).days <= 365)
        stale = sum(1 for day in dates if (today - day).days > 1095)
        if current * 3 >= len(dates) * 2:
            mix = "current-heavy"
        elif stale * 3 >= len(dates) * 2:
            mix = "stale-heavy"
        elif current and stale:
            mix = "mixed"
        else:
            mix = "mixed"
    acknowledges = bool(_ACK_RE.search(str(answer or "")))
    return {
        "recency_mix": mix,
        "acknowledges_recency_mix": acknowledges,
        "dated_citation_count": len(dates),
        "unknown_date_count": len(citations or []) - len(dates),
        "needs_acknowledgement": mix in {"mixed", "stale-heavy"},
        "passes": mix not in {"mixed", "stale-heavy"} or acknowledges,
    }
