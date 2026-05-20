from __future__ import annotations

from graph.rag.answer_claim_risk import analyze_answer_claim_risk


def test_answer_claim_risk_detects_claim_cues_and_uncited_paragraphs():
    report = analyze_answer_claim_risk(
        "Revenue was 42% higher in 2025 and always improves.\n\n"
        "This medical dose is safe [1].\n\n"
        "Plain caveat without a claim."
    )

    assert report["paragraph_count"] == 3
    assert report["claim_count"] == 2
    assert report["claim_type_counts"]["numeric_claim"] == 1
    assert report["claim_type_counts"]["date_claim"] == 1
    assert report["claim_type_counts"]["absolute_claim"] == 1
    assert report["claim_type_counts"]["high_stakes_domain"] == 1
    assert report["claim_type_counts"]["uncited_paragraph"] == 1
    assert report["risks"][0]["paragraph_index"] == 0
    assert report["risks"][0]["risk_level"] == "high"
    assert "high_stakes_claims" in report["warnings"]
