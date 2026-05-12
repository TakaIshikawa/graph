from __future__ import annotations

import pytest

from graph.rag.query_ambiguity import detect_query_ambiguity


def signal_types(result: dict) -> list[str]:
    return [signal["type"] for signal in result["signals"]]


def test_empty_query_returns_zero_score_and_no_terms():
    assert detect_query_ambiguity("   ") == {
        "normalized_query": "",
        "ambiguity_score": 0,
        "signals": [],
        "suggested_clarifying_terms": [],
    }


def test_unambiguous_query_has_no_signals():
    result = detect_query_ambiguity("battery degradation rate in 2024 field tests")

    assert result["normalized_query"] == "battery degradation rate in 2024 field tests"
    assert result["ambiguity_score"] == 0
    assert result["signals"] == []


def test_ambiguous_mixed_query_reports_stable_signals():
    result = detect_query_ambiguity(
        "What is its latest impact on Project Atlas and Acme? How should this change?"
    )

    assert signal_types(result) == [
        "broad_noun",
        "question_fanout",
        "relative_date",
        "unknown_capitalized_term",
        "vague_pronoun",
    ]
    assert result["ambiguity_score"] > 0.8
    assert result["signals"][0]["terms"] == ["impact", "project"]
    assert result["suggested_clarifying_terms"] == [
        "Acme",
        "impact",
        "its",
        "latest",
        "multiple_question_words",
        "project",
        "Project Atlas",
        "this",
    ]


def test_known_terms_suppress_unknown_capitalized_terms_case_insensitively():
    result = detect_query_ambiguity("Compare Project Atlas and Acme", known_terms=["project atlas", "ACME"])

    assert "unknown_capitalized_term" not in signal_types(result)
    assert signal_types(result) == ["question_fanout"]


def test_max_terms_limits_after_stable_sorting():
    result = detect_query_ambiguity("What about Foo Bar and Acme today this project?", max_terms=3)

    assert result["suggested_clarifying_terms"] == ["Acme", "Foo Bar", "project"]


@pytest.mark.parametrize("max_terms", [-1, 1.2, True, "3"])
def test_max_terms_validation_rejects_negative_or_non_integer_values(max_terms):
    with pytest.raises(ValueError, match="max_terms must be a non-negative integer"):
        detect_query_ambiguity("query", max_terms=max_terms)
