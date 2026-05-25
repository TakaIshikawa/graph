from __future__ import annotations

from graph.rag.answer_tradeoff_coverage import audit_answer_tradeoff_coverage


def test_reports_complete_tradeoff_coverage():
    report = audit_answer_tradeoff_coverage(
        "I recommend option A. Benefits improve speed, costs are low, risks are known, alternatives exist, and implementation effort is modest."
    )

    assert report["is_recommendation_answer"] is True
    assert all(report["coverage"].values())
    assert report["warnings"] == []


def test_flags_missing_categories_for_recommendation_queries():
    report = audit_answer_tradeoff_coverage("Use option A because it improves speed.", query="Which option should we choose?")

    assert report["missing_tradeoff_categories"] == ["costs", "risks", "alternatives", "effort"]
    assert "missing_costs" in report["warnings"]


def test_does_not_penalize_definitional_answers():
    report = audit_answer_tradeoff_coverage("A vector database stores embeddings for similarity search.")

    assert report["is_recommendation_answer"] is False
    assert report["missing_tradeoff_categories"] == []
