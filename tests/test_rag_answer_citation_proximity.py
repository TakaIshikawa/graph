from __future__ import annotations

from graph.rag.answer_citation_proximity import audit_answer_citation_proximity


def test_answer_citation_proximity_accepts_adjacent_citations():
    audit = audit_answer_citation_proximity("Revenue rose in 2024 [1]. Margin also improved.")

    assert audit["sentence_count"] == 2
    assert audit["cited_sentence_count"] == 1
    assert audit["unsupported_claim_sentences"] == []
    assert audit["proximity_score"] == 1.0


def test_answer_citation_proximity_supports_paragraph_end_citations():
    audit = audit_answer_citation_proximity("Revenue rose in 2024. Margin improved https://example.com/report")

    assert audit["cited_sentence_count"] == 1
    assert audit["unsupported_claim_sentences"] == []


def test_answer_citation_proximity_flags_uncited_claims():
    audit = audit_answer_citation_proximity("Revenue rose in 2024. Margin improved. Costs fell.")

    assert audit["unsupported_claim_sentences"] == [
        "Revenue rose in 2024.",
        "Margin improved.",
        "Costs fell.",
    ]
    assert audit["warnings"] == [
        "Claim sentence 1 has no nearby citation.",
        "Claim sentence 2 has no nearby citation.",
        "Claim sentence 3 has no nearby citation.",
    ]
