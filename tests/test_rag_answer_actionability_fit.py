from __future__ import annotations

from graph.rag.answer_actionability_fit import audit_answer_actionability_fit


def test_answer_actionability_fit_scores_actionable_answers():
    audit = audit_answer_actionability_fit(
        "How should we implement the migration?",
        "1. Inventory services. Team Alpha owns it by Monday. If errors exceed 1%, pause.",
    )

    assert audit["query_requires_action"] is True
    assert audit["missing_action_elements"] == []
    assert audit["actionability_score"] == 1.0


def test_answer_actionability_fit_flags_vague_answers():
    audit = audit_answer_actionability_fit("Recommend next steps", "The migration is important and should be handled carefully.")

    assert audit["missing_action_elements"] == ["steps", "owners", "dates", "decision criteria"]
    assert audit["actionability_score"] == 0.0


def test_answer_actionability_fit_ignores_non_action_queries():
    audit = audit_answer_actionability_fit("What is OAuth?", "OAuth is an authorization protocol.")

    assert audit["query_requires_action"] is False
    assert audit["missing_action_elements"] == []
    assert audit["actionability_score"] == 1.0
