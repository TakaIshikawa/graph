from __future__ import annotations

from graph.rag.answer_condition_coverage import audit_answer_condition_coverage


def test_detects_answer_condition_types():
    report = audit_answer_condition_coverage("Use it only when traffic is low unless the cache is warm; otherwise fall back.")

    assert report["has_condition_cues"] is True
    assert report["answer_condition_types"] == ["exception", "fallback", "prerequisite"]


def test_flags_unconditional_recommendation_when_evidence_has_conditions():
    report = audit_answer_condition_coverage(
        "You should use the new policy.",
        evidence=[{"content": "The policy applies only when users opt in, except regulated accounts."}],
    )

    assert report["missing_condition_types"] == ["boundary", "exception", "prerequisite"]
    assert report["warnings"] == ["unconditional_recommendation_with_conditional_evidence"]
    assert len(report["matched_evidence_snippets"]) == 1


def test_no_conditions_is_stable():
    report = audit_answer_condition_coverage("The answer summarizes the findings.")

    assert report["missing_condition_types"] == []
    assert report["matched_evidence_snippets"] == []
