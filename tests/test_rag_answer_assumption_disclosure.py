from __future__ import annotations

from graph.rag.answer_assumption_disclosure import audit_answer_assumption_disclosure


def test_identifies_explicit_assumption_section():
    result = audit_answer_assumption_disclosure(
        "Estimate next year's costs.",
        "Assumptions: usage grows 10%. The estimate is $12k.",
    )

    assert result["explicit_assumption_section"] is True
    assert result["has_assumption_disclosure"] is True
    assert result["missing_assumption_risk"] < 0.3


def test_identifies_inline_assumption_phrase():
    result = audit_answer_assumption_disclosure(
        "Compare the options.",
        "The cloud option is cheaper if we assume storage remains flat.",
    )

    assert "if we assume" in result["inline_assumption_phrases"]
    assert result["has_assumption_disclosure"] is True


def test_forecast_without_assumptions_has_high_risk():
    result = audit_answer_assumption_disclosure("Forecast revenue.", "Revenue will be $5M.")

    assert result["query_needs_assumptions"] is True
    assert result["risk_level"] == "high"


def test_empty_and_neutral_answers_are_deterministic():
    empty = audit_answer_assumption_disclosure("What is TLS?", "")
    neutral = audit_answer_assumption_disclosure("What is TLS?", "TLS encrypts network traffic.")

    assert empty["has_assumption_disclosure"] is False
    assert empty["risk_level"] == "low"
    assert neutral["missing_assumption_risk"] == 0.1
