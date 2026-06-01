"""Audit stale citation use when answer text omits recency disclosure."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Any

from graph.rag._analysis_utils import coerce_now, parse_date, result_id, value

_DISCLOSURE_RE = re.compile(r"\b(old|older|stale|outdated|as of|from \d{4}|published in \d{4}|source age)\b", re.I)


def audit_answer_citation_recency_disclosure(
    answer: str, citations: Iterable[Any], *, current_date: Any = None, stale_after_days: int = 730
) -> list[dict[str, Any]]:
    if _DISCLOSURE_RE.search(str(answer or "")):
        return []
    today = coerce_now(current_date)
    rows = []
    for index, citation in enumerate(citations or []):
        cited = _citation_date(citation)
        if cited is None:
            continue
        age = (today - cited).days
        if age > stale_after_days:
            rows.append({"citation_id": result_id(citation, index), "citation_date": cited.isoformat(), "age_days": age, "severity": "medium"})
    return sorted(rows, key=lambda row: (row["citation_date"], row["citation_id"]))


def _citation_date(citation: Any) -> date | None:
    for key in ("publication_date", "published_at", "date", "year"):
        parsed = parse_date(value(citation, key))
        if parsed:
            return parsed
    return None
