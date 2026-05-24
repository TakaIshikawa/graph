from __future__ import annotations

from graph.rag.answer_uncertainty_quantification import audit_answer_uncertainty_quantification


def test_detects_numeric_ranges_and_probabilities():
    result = audit_answer_uncertainty_quantification(
        "Estimate migration duration.",
        "It will likely take 6 to 8 weeks with a 70% chance of finishing on time.",
    )

    assert result["has_quantified_uncertainty"] is True
    assert result["markers"]["numeric_ranges"] == ["6 to 8 weeks"]
    assert result["markers"]["probabilities"] == ["70% chance"]


def test_detects_confidence_intervals():
    result = audit_answer_uncertainty_quantification("Estimate effect size.", "The 95% CI was 1.2 to 1.8.")

    assert result["has_quantified_uncertainty"] is True
    assert result["markers"]["confidence_intervals"] == ["CI was 1.2 to"]


def test_reports_missing_quantification_for_predictive_claim():
    result = audit_answer_uncertainty_quantification("Forecast demand.", "Demand will rise next year.")

    assert result["query_needs_uncertainty"] is True
    assert result["missing_quantification"] is True


def test_ordinary_numeric_fact_is_not_uncertainty():
    result = audit_answer_uncertainty_quantification("What is the port?", "HTTPS usually uses port 443.")

    assert result["has_uncertainty_marker"] is False
    assert result["missing_quantification"] is False
