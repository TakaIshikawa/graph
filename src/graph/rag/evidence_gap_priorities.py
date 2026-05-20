"""Prioritize heterogeneous evidence gap records for follow-up retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import MISSING, result_id, string, value

_SEVERITY = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


def prioritize_evidence_gaps(gaps: Iterable[Any], *, max_items: int = 10) -> dict[str, Any]:
    """Normalize and sort evidence gaps by retrieval priority."""
    if not isinstance(max_items, int) or isinstance(max_items, bool) or max_items < 1:
        raise ValueError("max_items must be a positive integer")
    records = [_record(gap, index) for index, gap in enumerate(gaps)]
    records.sort(key=lambda item: (-item["priority_score"], _sort_key(item["gap_id"])))
    return {"total_gaps": len(records), "priorities": records[:max_items]}


def _record(gap: Any, index: int) -> dict[str, Any]:
    gap_id = result_id(gap, index)
    severity = _severity(_first_value(gap, ("severity", "level")))
    missing = string(_first_value(gap, ("missing_field", "field", "gap_type"))) or "evidence"
    source = string(_first_value(gap, ("source", "source_project", "source_id")))
    claim = string(_first_value(gap, ("claim", "statement")))
    action = _action(missing, source, claim)
    impact = (2 if claim else 0) + (2 if missing in {"citation", "source"} else 0) + (1 if "date" in missing else 0)
    score = _SEVERITY[severity] * 10 + impact
    reasons = [f"{severity}_severity", f"missing_{missing}"]
    if claim:
        reasons.append("claim_impact")
    if source:
        reasons.append("source_specific")
    return {
        "gap_id": gap_id,
        "severity": severity,
        "missing_field": missing,
        "source": source,
        "claim": claim,
        "priority_score": score,
        "reasons": reasons,
        "recommended_action": action,
    }


def _severity(value_: Any) -> str:
    text = (string(value_) or "medium").casefold()
    if text in _SEVERITY:
        return text
    if text in {"blocker", "severe"}:
        return "critical"
    if text in {"warning", "warn"}:
        return "medium"
    return "medium"


def _action(missing: str, source: str | None, claim: str | None) -> str:
    if "date" in missing:
        return "find_date_support"
    if missing == "source" or source:
        return "retrieve_primary_source"
    if missing == "citation" or claim:
        return "add_citation"
    return "retrieve_supporting_evidence"


def _first_value(gap: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        item = value(gap, key)
        if item is not MISSING and item is not None and string(item) is not None:
            return item
    return MISSING


def _sort_key(value_: object) -> tuple[str, str]:
    text = "" if value_ is None else str(value_)
    return (text.casefold(), text)
