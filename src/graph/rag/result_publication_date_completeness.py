"""Analyze publication date coverage in retrieval results."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id, string, value

_DATE_KEYS = ("published_at", "publication_date", "published", "updated_at", "updated", "accessed_at", "accessed", "date")
_TEXT_DATE_RE = re.compile(
    r"\b(?:published|updated|accessed)\s+(?:on\s+)?(?:\d{4}-\d{1,2}-\d{1,2}|(?:19|20)\d{2}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b",
    re.I,
)


def analyze_result_publication_date_completeness(results: Iterable[Any]) -> dict[str, Any]:
    rows = list(results or [])
    missing = []
    with_dates = 0
    for index, result in enumerate(rows):
        if _has_date_signal(result):
            with_dates += 1
        else:
            missing.append(result_id(result, index))
    return {"total_results": len(rows), "results_with_dates": with_dates, "results_missing_dates": len(missing), "missing_date_result_ids": missing}


def _has_date_signal(result: Any) -> bool:
    for key in _DATE_KEYS:
        if string(value(result, key)):
            return True
    return bool(_TEXT_DATE_RE.search(content_text(result)))
