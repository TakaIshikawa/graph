from __future__ import annotations

from graph.rag.answer_assumption_audit import audit_answer_assumptions


def test_answer_assumption_audit_flags_core_cues():
    report = audit_answer_assumptions(
        "The outage happened because cache pressure increased. "
        "The user wants to reduce latency. "
        "You should switch providers because it is always faster."
    )

    codes = {item["code"] for item in report["assumptions"]}
    assert {"unstated_causality", "inferred_intent", "unsupported_recommendation", "generalized_scope"} <= codes
    assert report["warnings"] == ["assumptions_detected"]


def test_answer_assumption_audit_evidence_overlap_reduces_severity():
    report = audit_answer_assumptions("This may be caused by cache pressure.", evidence="cache pressure caused retries")

    assert report["assumptions"][0]["support_signal"] == "evidence_overlap"
    assert report["assumptions"][0]["severity"] == "low"


def test_answer_assumption_audit_neutral_and_empty_answers():
    assert audit_answer_assumptions("The report lists three options.")["assumption_count"] == 0
    assert audit_answer_assumptions("")["assumptions"] == []
