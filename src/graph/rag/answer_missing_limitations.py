"""Recommend answer limitations when evidence has support risks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import coerce_now, number, result_date, source_id, value


def audit_answer_missing_limitations(answer: str, evidence: Iterable[Any], *, now: Any = None) -> dict[str, Any]:
    """Detect missing caveats for sparse, stale, narrow, or low-confidence evidence."""
    rows = list(evidence)
    today = coerce_now(now)
    risks: list[str] = []
    if len(rows) < 2:
        risks.append("sparse evidence")
    sources = {source_id(row) for row in rows if source_id(row)}
    if rows and len(sources) < 2:
        risks.append("single-source evidence")
    dates = [parsed for row in rows if (parsed := result_date(row)) is not None]
    if rows and (not dates or all((today - parsed).days > 365 for parsed in dates)):
        risks.append("old evidence dates")
    if any((score := number(value(row, "confidence"))) is not None and score < 0.5 for row in rows):
        risks.append("low confidence metadata")

    present = _present_limitations(answer)
    missing = [risk for risk in risks if risk not in present]
    caveats = [_caveat(risk) for risk in missing]
    score = 1.0 if not risks else round((len(risks) - len(missing)) / len(risks), 2)
    return {
        "missing_limitations": missing,
        "evidence_risks": risks,
        "recommended_caveats": caveats,
        "limitation_score": score,
    }


def _present_limitations(answer: str) -> set[str]:
    text = (answer or "").casefold()
    present = set()
    if "limited evidence" in text or "sparse" in text:
        present.add("sparse evidence")
    if "single source" in text or "one source" in text:
        present.add("single-source evidence")
    if "stale" in text or "old evidence" in text:
        present.add("old evidence dates")
    if "low confidence" in text:
        present.add("low confidence metadata")
    return present


def _caveat(risk: str) -> str:
    return {
        "sparse evidence": "State that the answer is based on limited evidence.",
        "single-source evidence": "Note that corroborating sources were not found.",
        "old evidence dates": "Mention that fresher evidence may change the conclusion.",
        "low confidence metadata": "Disclose that some evidence is marked low confidence.",
    }[risk]
