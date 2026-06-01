from graph.rag.query_cost_constraint import detect_query_cost_constraints


def test_detects_explicit_budget_constraints():
    rows = detect_query_cost_constraints("Keep it within budget, under $500, and include license costs.")

    assert rows == [
        {"matched_text": "within budget", "category": "budget"},
        {"matched_text": "under $500", "category": "cost_cap"},
        {"matched_text": "license costs", "category": "license_cost"},
    ]


def test_detects_price_comparison_token_cost_and_tco_wording():
    rows = detect_query_cost_constraints("Compare pricing, token cost, and total cost of ownership.")

    assert rows == [
        {"matched_text": "pricing", "category": "pricing"},
        {"matched_text": "total cost of ownership", "category": "tco"},
        {"matched_text": "token cost", "category": "token_cost"},
    ]


def test_cost_constraints_deduplicate_and_ignore_unrelated_metaphors():
    assert detect_query_cost_constraints("Budget and spending limit should be listed.") == [
        {"matched_text": "Budget", "category": "budget"}
    ]
    assert detect_query_cost_constraints("What is the human cost of delaying the launch?") == []
