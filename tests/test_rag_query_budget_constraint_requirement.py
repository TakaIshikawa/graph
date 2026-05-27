from __future__ import annotations

from graph.rag.query_budget_constraint_requirement import detect_query_budget_constraint_requirement


def test_detect_query_budget_constraint_requirement_extracts_currency_amounts():
    result = detect_query_budget_constraint_requirement("Find options with a cost ceiling under $250 per month.")

    assert result["requires_budget_constraint"] is True
    assert result["constraint_terms"] == ["price", "cost_ceiling"]
    assert result["budget_values"] == ["$250"]


def test_detect_query_budget_constraint_requirement_extracts_token_limits():
    result = detect_query_budget_constraint_requirement("Keep the answer within a 8k token budget.")

    assert result["requires_budget_constraint"] is True
    assert result["constraint_terms"] == ["budget", "token_budget"]
    assert result["budget_values"] == ["8k token"]


def test_detect_query_budget_constraint_requirement_extracts_time_limits():
    result = detect_query_budget_constraint_requirement("Use sources that let us respond under 30 seconds.")

    assert result["requires_budget_constraint"] is True
    assert result["constraint_terms"] == ["time_budget"]
    assert result["budget_values"] == ["30 seconds"]


def test_detect_query_budget_constraint_requirement_detects_free_tier_wording():
    result = detect_query_budget_constraint_requirement("Compare only tools that work on the free tier or starter plan.")

    assert result["requires_budget_constraint"] is True
    assert result["constraint_terms"] == ["subscription_tier", "free_or_paid"]
    assert result["budget_values"] == ["free tier", "starter plan"]


def test_detect_query_budget_constraint_requirement_ignores_non_budget_queries():
    result = detect_query_budget_constraint_requirement("Summarize the evidence about model accuracy by dataset.")

    assert result == {
        "requires_budget_constraint": False,
        "constraint_terms": [],
        "budget_values": [],
        "rationale": "No budget, pricing, token, tier, or time-budget constraints were detected.",
    }
