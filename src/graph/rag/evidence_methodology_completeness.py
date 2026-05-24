"""Audit methodology signal completeness in evidence snippets."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, result_id

_SIGNALS: dict[str, re.Pattern[str]] = {
    "sample": re.compile(r"\b(?:sample|n\s*=|participants?|respondents?|subjects?|cases?)\b", re.I),
    "population": re.compile(r"\b(?:population|adults|children|patients|users|households|companies|cohort)\b", re.I),
    "measurement": re.compile(r"\b(?:measured|metric|outcome|rate|score|survey|instrument|endpoint)\b", re.I),
    "timeframe": re.compile(r"\b(?:19|20)\d{2}\b|\b(?:week|month|year|quarter|during|between|from|through)\b", re.I),
    "limitations": re.compile(r"\b(?:limit(?:ation)?s?|bias|uncertain|caveat|self-reported|small sample|not representative)\b", re.I),
}


def audit_evidence_methodology_completeness(results: Iterable[Any]) -> dict[str, Any]:
    """Return per-result methodology rows and aggregate missing signal counts."""
    rows = []
    missing_counts: Counter[str] = Counter()
    for index, result in enumerate(results):
        text = content_text(result)
        present = [name for name, pattern in _SIGNALS.items() if pattern.search(text)]
        missing = [name for name in _SIGNALS if name not in present]
        missing_counts.update(missing)
        rows.append(
            {
                "result_id": result_id(result, index),
                "signals_present": present,
                "signals_missing": missing,
                "completeness_score": round(len(present) / len(_SIGNALS), 2),
            }
        )
    return {"rows": rows, "missing_signal_counts": dict(sorted(missing_counts.items()))}
