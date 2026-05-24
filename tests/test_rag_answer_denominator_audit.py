from __future__ import annotations

from graph.rag.answer_denominator_audit import audit_answer_denominators


def test_answer_denominator_audit_flags_missing_fields_separately():
    audit = audit_answer_denominators("Conversion rose 42%. The average was higher.")

    assert audit["total_claims"] == 2
    first = audit["claims"][0]
    assert first["span"] == "42%"
    assert "missing_denominator" in first["missing_fields"]
    assert "missing_population" in first["missing_fields"]
    assert "missing_timeframe" in first["missing_fields"]
    assert "missing_unit" not in first["missing_fields"]
    assert audit["warnings"] == ["ambiguous_quantitative_claims"]


def test_answer_denominator_audit_allows_nearby_denominator_phrases():
    audit = audit_answer_denominators(
        "In 2025, 42% of 200 users completed the flow during the trial period."
    )

    assert audit["claims"][0]["missing_fields"] == []
    assert audit["claims"][0]["status"] == "supported"
    assert audit["ambiguous_claims"] == 0


def test_answer_denominator_audit_returns_empty_for_no_quantitative_claims():
    assert audit_answer_denominators("The result improved.")["claims"] == []
