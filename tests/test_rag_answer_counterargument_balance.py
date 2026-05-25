from __future__ import annotations

from graph.rag.answer_counterargument_balance import audit_answer_counterargument_balance


def test_counterargument_balance_flags_one_sided_recommendations():
    result = audit_answer_counterargument_balance("Should we choose A or B?", "Choose A because it is cheaper.")

    assert result["requires_counterargument"] is True
    assert result["missing_counterargument"] is True
    assert result["balance_score"] == 0.0


def test_counterargument_balance_recognizes_limitation_cues():
    result = audit_answer_counterargument_balance("Compare A vs B.", "A is cheaper. However, B has fewer operational risks.")

    assert result["missing_counterargument"] is False
    assert "however" in result["matched_balance_cues"]


def test_counterargument_balance_does_not_require_balance_for_factual_queries():
    result = audit_answer_counterargument_balance("What is the capital of France?", "Paris is the capital.")

    assert result["requires_counterargument"] is False
    assert result["missing_counterargument"] is False
