from __future__ import annotations

from graph.rag.query_cost_sensitivity_requirement import detect_query_cost_sensitivity_requirement


def test_detect_query_cost_sensitivity_requirement_extracts_budget_amounts():
    result = detect_query_cost_sensitivity_requirement("Find tools under $50 and less than 20 EUR per seat.")

    assert result["requires_cost_awareness"] is True
    assert result["cost_mode"] == "budget_capped"
    assert result["budget_amounts"] == [
        {"amount": "$50", "cue": "under $50", "span": [11, 20]},
        {"amount": "20 EUR", "cue": "less than 20 EUR", "span": [25, 41]},
    ]
    assert result["currency_mentions"] == ["$", "EUR"]


def test_detect_query_cost_sensitivity_requirement_maps_common_cost_modes():
    assert detect_query_cost_sensitivity_requirement("Only free open-source options.")["cost_mode"] == "free"
    assert detect_query_cost_sensitivity_requirement("Prefer cheap hosted products.")["cost_mode"] == "low_cost"
    assert detect_query_cost_sensitivity_requirement("Include enterprise pricing and total cost.")["cost_mode"] == "total_cost_aware"


def test_detect_query_cost_sensitivity_requirement_returns_none_without_cost_language():
    assert detect_query_cost_sensitivity_requirement("Compare retrieval accuracy by dataset.") == {
        "requires_cost_awareness": False,
        "cost_mode": "none",
        "budget_amounts": [],
        "currency_mentions": [],
        "cues": [],
    }
