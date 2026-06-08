from __future__ import annotations

from graph.rag.query_slo_error_budget_requirement import detect_query_slo_error_budget_requirement


def test_detects_slo_error_budget_categories():
    result = detect_query_slo_error_budget_requirement(
        "Define SLO, error budget, burn rate, SLI, reliability objective, and alert threshold requirements."
    )

    assert result["has_slo_error_budget_requirement"] is True
    assert result["requirements"] == [
        {"category": "alert_threshold", "matched_text": "alert threshold"},
        {"category": "burn_rate", "matched_text": "burn rate"},
        {"category": "error_budget", "matched_text": "error budget"},
        {"category": "reliability_objective", "matched_text": "reliability objective"},
        {"category": "sli", "matched_text": "SLI"},
        {"category": "slo", "matched_text": "SLO"},
    ]


def test_pure_sla_credit_query_does_not_match():
    assert detect_query_slo_error_budget_requirement(
        "Explain SLA service credits and refund penalties for downtime."
    ) == {"has_slo_error_budget_requirement": False, "requirements": []}
