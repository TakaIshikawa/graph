from __future__ import annotations

import pytest

from graph.rag.query_regulatory_constraints import detect_query_regulatory_constraints


def test_regulatory_constraints_no_match_has_zero_confidence():
    result = detect_query_regulatory_constraints("Explain semantic search ranking.")

    assert result["frameworks"] == []
    assert result["jurisdiction_hints"] == []
    assert result["required_source_classes"] == []
    assert result["confidence"] == 0.0
    assert result["rationale"] == ["no_regulatory_cues"]


def test_regulatory_constraints_detects_single_framework_and_jurisdiction():
    result = detect_query_regulatory_constraints("What HIPAA rules apply to PHI in the United States?")

    assert result["frameworks"] == [
        {
            "framework": "HIPAA",
            "matched_cues": ["hipaa", "health privacy"],
            "required_source_classes": ["regulator guidance", "official compliance text", "legal analysis"],
        }
    ]
    assert result["jurisdiction_hints"] == ["United States"]
    assert "official compliance text" in result["required_source_classes"]
    assert result["confidence"] == 0.65
    assert result["rationale"] == ["matched_framework:HIPAA", "jurisdiction_hint:United States"]


def test_regulatory_constraints_detects_multiple_named_frameworks():
    result = detect_query_regulatory_constraints(
        "Compare GDPR, SEC disclosure, OSHA, FDA, and SOC 2 compliance obligations for EU and US teams."
    )

    assert [match["framework"] for match in result["frameworks"]] == ["GDPR", "SEC", "OSHA", "FDA", "SOC 2"]
    assert result["jurisdiction_hints"] == ["United States", "European Union"]
    assert result["confidence"] == 0.95


@pytest.mark.parametrize("query", ["", "  ", None])
def test_regulatory_constraints_validates_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        detect_query_regulatory_constraints(query)  # type: ignore[arg-type]
