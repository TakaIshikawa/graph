from __future__ import annotations

from graph.rag.answer_citation_placement import audit_answer_citation_placement


def test_audits_cited_and_uncited_claim_sentences():
    report = audit_answer_citation_placement(
        "Revenue rose 42% in 2025 [1]. "
        "Medical dose claims are always safe. "
        "Plain caveat."
    )

    assert report["sentence_count"] == 3
    assert report["claim_sentence_count"] == 2
    assert report["cited_claim_count"] == 1
    assert report["uncited_claim_count"] == 1
    assert report["claim_type_counts"]["numeric_claim"] == 1
    assert report["claim_type_counts"]["date_claim"] == 1
    assert report["claim_type_counts"]["absolute_claim"] == 1
    assert report["claim_type_counts"]["high_stakes_domain"] == 1
    assert "uncited_absolute_claim" in report["warnings"]
    assert "uncited_high_stakes_domain" in report["warnings"]


def test_custom_citation_patterns_count_as_cited():
    report = audit_answer_citation_placement("The contract changed in 2024 {src:A}.", citation_patterns=[r"\{src:[^}]+\}"])

    assert report["uncited_claim_count"] == 0
    assert report["sentences"][0]["citation_status"] == "cited"


def test_empty_answer_returns_no_answer_warning():
    report = audit_answer_citation_placement("  ")

    assert report["sentence_count"] == 0
    assert report["claim_sentence_count"] == 0
    assert report["warnings"] == ["no_answer"]
