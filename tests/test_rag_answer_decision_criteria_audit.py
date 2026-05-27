from __future__ import annotations

from graph.rag.answer_decision_criteria_audit import audit_answer_decision_criteria


def test_audit_answer_decision_criteria_detects_criteria_thresholds_and_tradeoffs():
    result = audit_answer_decision_criteria(
        "Choose if uptime is the priority. Must have at least 99.9% availability. "
        "The tradeoff is higher cost. Tie-breaker: prefer the simpler vendor."
    )

    assert result["criteria_count"] == 2
    assert result["threshold_count"] == 2
    assert result["tradeoff_count"] == 2
    assert result["missing_decision_elements"] == []
    assert result["decision_readiness_score"] == 1.0


def test_audit_answer_decision_criteria_lists_missing_elements_for_decision_answers():
    result = audit_answer_decision_criteria("I recommend option A because it is easier to deploy.")

    assert result["criteria_count"] == 1
    assert result["threshold_count"] == 0
    assert result["missing_decision_elements"] == ["thresholds", "tie_breakers"]
    assert result["decision_readiness_score"] == 0.333


def test_audit_answer_decision_criteria_does_not_penalize_neutral_explanations():
    assert audit_answer_decision_criteria("The system stores chunks, embeddings, and source metadata.") == {
        "criteria_count": 0,
        "threshold_count": 0,
        "tradeoff_count": 0,
        "missing_decision_elements": [],
        "decision_readiness_score": 1.0,
    }
