from __future__ import annotations

from graph.rag.answer_limitation_disclosure import audit_answer_limitation_disclosure


def test_classifies_limitation_types():
    result = audit_answer_limitation_disclosure(
        "Compare all vendors.",
        "Limited evidence is available, and the data is unavailable for smaller vendors. Results may not apply overseas.",
    )

    assert result["has_limitation_disclosure"] is True
    assert result["limitation_types"] == ["incomplete_evidence", "unavailable_data", "applicability_constraints"]


def test_broad_query_without_limitations_scores_lower():
    result = audit_answer_limitation_disclosure("What is the best option across regions?", "Option A is best.")

    assert result["broad_or_comparative_query"] is True
    assert result["disclosure_score"] == 0.25


def test_hedging_without_named_limitation_is_not_disclosure():
    result = audit_answer_limitation_disclosure("Compare the options.", "This might be the best answer.")

    assert result["has_limitation_disclosure"] is False
    assert result["hedge_without_named_limitation"] is True


def test_outdated_and_scope_limitations():
    result = audit_answer_limitation_disclosure(
        "Summarize the evidence.",
        "The review is limited to US trials and may include outdated sources.",
    )

    assert result["limitation_types"] == ["narrow_scope", "outdated_sources"]
